"""FeitCSI .dat parser. Wraps CSIKit to return amplitude/phase per frame."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from CSIKit.reader import FeitCSIBeamformReader
from CSIKit.util import csitools


def mimo_safe_interpolate(upstream: Callable[[dict], dict], csi: dict) -> dict:
    """Pilot interpolation that survives multi-stream frames.

    CSIKit's interpolate() loops over the rx axis of a (subcarrier, rx, tx)
    matrix, so its wrap-correction test `phase[i-1, j] > 2` is a bare `if` on
    an array of length num_tx. That is only unambiguous while num_tx == 1;
    a 2x2 frame raises ValueError and kills the whole read.

    Interpolation is independent per stream, so folding tx into the rx axis
    gives each (rx, tx) pair its own j and restores the scalar comparison
    upstream assumes. Same arithmetic, same results, no forked index table.
    """
    matrix = csi["csi_matrix"]
    shape = matrix.shape

    if matrix.ndim != 3 or shape[2] <= 1:
        return upstream(csi)

    csi["csi_matrix"] = matrix.reshape(shape[0], shape[1] * shape[2], 1)
    csi = upstream(csi)
    csi["csi_matrix"] = csi["csi_matrix"].reshape(shape)
    return csi


@dataclass(frozen=True)
class FeitCSICapture:
    amplitude: np.ndarray
    phase: np.ndarray
    time_seconds: np.ndarray
    bandwidth: str
    chipset: str
    filename: str
    num_subcarriers: int

    def __len__(self) -> int:
        return int(self.amplitude.shape[0])


def load_capture(
    path: str | Path,
    *,
    scaled: bool = True,
    interpolate: bool = True,
    filter_mac: str | None = None,
) -> FeitCSICapture:
    """Full-file parse via CSIKit. Reference path; see stream.CaptureStream.

    Subcarriers are returned in the order FeitCSI emits them, which is already
    centred (index 0 is the lowest subcarrier, index N//2 is DC). Do not
    fftshift: the array is not in FFT bin order, so shifting it splits a
    contiguous spectrum and welds the two outer edges together. Measured on an
    80 MHz capture, that injects a 5.5 dB discontinuity at the seam where the
    largest genuine bin-to-bin step is 0.34 dB, and relabels DC as bin -N//2.
    """
    path = Path(path)
    reader = FeitCSIBeamformReader()

    if interpolate:
        _upstream = reader.interpolate
        reader.interpolate = lambda csi: mimo_safe_interpolate(_upstream, csi)

    csi_data = reader.read_file(
        str(path),
        scaled=scaled,
        remove_unusable_subcarriers=True,
        filter_mac=filter_mac,
        interpolate=interpolate,
    )

    amplitude, _, _ = csitools.get_CSI(
        csi_data, metric="amplitude", extract_as_dBm=True, squeeze_output=True
    )
    phase, _, _ = csitools.get_CSI(
        csi_data, metric="phase", extract_as_dBm=False, squeeze_output=True
    )
    time_seconds = np.asarray(csi_data.timestamps, dtype=float)

    # csitools.get_CSI returns (frames, subcarriers, rx, tx). squeeze_output
    # only drops rx/tx axes for SISO. Force SISO slice for MIMO captures so
    # downstream code can assume 2D (frames, subcarriers).
    while amplitude.ndim > 2:
        amplitude = amplitude[..., 0]
    while phase.ndim > 2:
        phase = phase[..., 0]

    amplitude = np.atleast_2d(amplitude)
    phase = np.atleast_2d(phase)

    bandwidth = str(csi_data.bandwidth) if csi_data.bandwidth is not None else "unknown"
    chipset = str(csi_data.chipset) if csi_data.chipset else "unknown"

    return FeitCSICapture(
        amplitude=amplitude,
        phase=phase,
        time_seconds=time_seconds,
        bandwidth=bandwidth,
        chipset=chipset,
        filename=path.name,
        num_subcarriers=int(amplitude.shape[1]),
    )


def tail_window(
    capture: FeitCSICapture,
    *,
    max_packets: int = 200,
    start_time: float | None = None,
) -> FeitCSICapture:
    if len(capture) == 0:
        return capture

    amp = capture.amplitude
    phase = capture.phase
    t = capture.time_seconds

    if start_time is not None and t.size > 0:
        mask = t >= start_time
        if not mask.any():
            return FeitCSICapture(
                amplitude=amp[0:0],
                phase=phase[0:0],
                time_seconds=t[0:0],
                bandwidth=capture.bandwidth,
                chipset=capture.chipset,
                filename=capture.filename,
                num_subcarriers=capture.num_subcarriers,
            )
        amp = amp[mask]
        phase = phase[mask]
        t = t[mask]

    if max_packets > 0 and amp.shape[0] > max_packets:
        amp = amp[-max_packets:]
        phase = phase[-max_packets:]
        t = t[-max_packets:]

    return FeitCSICapture(
        amplitude=amp,
        phase=phase,
        time_seconds=t,
        bandwidth=capture.bandwidth,
        chipset=capture.chipset,
        filename=capture.filename,
        num_subcarriers=capture.num_subcarriers,
    )
