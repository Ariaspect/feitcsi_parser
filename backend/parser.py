"""FeitCSI .dat parser. Wraps CSIKit to return amplitude/phase per frame."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from CSIKit.reader import FeitCSIBeamformReader
from CSIKit.util import csitools
from CSIKit.util.matlab import db


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
    ratio_amplitude: np.ndarray
    ratio_phase: np.ndarray
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

    # squeeze_output=False keeps (frames, subcarriers, rx, tx) so we can
    # access rx1 for the CSI ratio. The existing code squeezed then took
    # [..., 0, 0]; indexing the unsqueezed array is equivalent for rx0tx0.
    amplitude_full, _, _ = csitools.get_CSI(
        csi_data, metric="amplitude", extract_as_dBm=True, squeeze_output=False
    )
    phase_full, _, _ = csitools.get_CSI(
        csi_data, metric="phase", extract_as_dBm=False, squeeze_output=False
    )
    time_seconds = np.asarray(csi_data.timestamps, dtype=float)

    amplitude = amplitude_full[..., 0, 0]
    phase = phase_full[..., 0, 0]

    # CSI ratio: rx1/rx0 (same tx0). Reconstruct complex CSI from amplitude
    # (dB, 20*log10) and phase, then divide — matching batch.py's complex
    # division so the ±π phase wrapping is identical across both paths.
    # 10**(dB/20) inverts db(|csi|)=20*log10(|csi|) back to |csi|. dbinv
    # cannot be used here: it is 10^(x/10), the inverse of 10*log10 (power),
    # not 20*log10 (amplitude).
    num_rx = amplitude_full.shape[-2] if amplitude_full.ndim >= 4 else 1
    if num_rx >= 2:
        csi_rx0 = 10 ** (amplitude_full[..., 0, 0] / 20) * np.exp(1j * phase_full[..., 0, 0])
        csi_rx1 = 10 ** (amplitude_full[..., 1, 0] / 20) * np.exp(1j * phase_full[..., 1, 0])
        ratio = csi_rx1 / csi_rx0
        ratio_amplitude = db(np.abs(ratio))
        ratio_phase = np.angle(ratio)
    else:
        ratio_amplitude = np.full_like(amplitude, np.nan)
        ratio_phase = np.full_like(phase, np.nan)

    amplitude = np.atleast_2d(amplitude)
    phase = np.atleast_2d(phase)
    ratio_amplitude = np.atleast_2d(ratio_amplitude)
    ratio_phase = np.atleast_2d(ratio_phase)

    bandwidth = str(csi_data.bandwidth) if csi_data.bandwidth is not None else "unknown"
    chipset = str(csi_data.chipset) if csi_data.chipset else "unknown"

    return FeitCSICapture(
        amplitude=amplitude,
        phase=phase,
        ratio_amplitude=ratio_amplitude,
        ratio_phase=ratio_phase,
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
    ratio_amp = capture.ratio_amplitude
    ratio_phase = capture.ratio_phase
    t = capture.time_seconds

    if start_time is not None and t.size > 0:
        mask = t >= start_time
        if not mask.any():
            return FeitCSICapture(
                amplitude=amp[0:0],
                phase=phase[0:0],
                ratio_amplitude=ratio_amp[0:0],
                ratio_phase=ratio_phase[0:0],
                time_seconds=t[0:0],
                bandwidth=capture.bandwidth,
                chipset=capture.chipset,
                filename=capture.filename,
                num_subcarriers=capture.num_subcarriers,
            )
        amp = amp[mask]
        phase = phase[mask]
        ratio_amp = ratio_amp[mask]
        ratio_phase = ratio_phase[mask]
        t = t[mask]

    if max_packets > 0 and amp.shape[0] > max_packets:
        amp = amp[-max_packets:]
        phase = phase[-max_packets:]
        ratio_amp = ratio_amp[-max_packets:]
        ratio_phase = ratio_phase[-max_packets:]
        t = t[-max_packets:]

    return FeitCSICapture(
        amplitude=amp,
        phase=phase,
        ratio_amplitude=ratio_amp,
        ratio_phase=ratio_phase,
        time_seconds=t,
        bandwidth=capture.bandwidth,
        chipset=capture.chipset,
        filename=capture.filename,
        num_subcarriers=capture.num_subcarriers,
    )
