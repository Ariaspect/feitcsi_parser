"""FeitCSI .dat parser wrapping CSIKit.

FeitCSI captures CSI on Intel AX200/AX210 NICs. Files are binary:
sequence of (272-byte header + CSI block) records. CSIKit reads the
whole file at once; this module provides a thin dataclass wrapper plus
rolling-window helpers for realtime visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from CSIKit.reader import FeitCSIBeamformReader
from CSIKit.util import csitools


@dataclass(frozen=True)
class FeitCSICapture:
    """Parsed FeitCSI capture.

    Attributes
    ----------
    amplitude : np.ndarray
        Shape (frames, subcarriers) dBm amplitude (SISO squeezed).
    phase : np.ndarray
        Shape (frames, subcarriers) phase in radians, wrapped [-pi, pi].
    time_seconds : np.ndarray
        Relative seconds per frame (frame 0 == 0.0). Derived from ftm_clock.
    bandwidth : str
        Bandwidth label from CSIKit (e.g. "80").
    chipset : str
        NIC chipset (e.g. "Intel AX2xx").
    filename : str
        Source file basename.
    num_subcarriers : int
        Subcarriers per frame.
    """

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
    fftshift: bool = True,
) -> FeitCSICapture:
    """Load FeitCSI .dat file via CSIKit.

    Parameters
    ----------
    path : str | Path
        Path to .dat file.
    scaled : bool
        Apply RSSI-based scaling (dBm amplitude).
    interpolate : bool
        Replace pilot subcarriers via interpolation.
    filter_mac : str | None
        Optional source MAC filter "AA:BB:CC:DD:EE:FF".
    fftshift : bool
        Apply np.fft.fftshift on subcarrier axis (signed frequency order).

    Returns
    -------
    FeitCSICapture
    """
    path = Path(path)
    reader = FeitCSIBeamformReader()
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

    amplitude = np.atleast_2d(amplitude)
    phase = np.atleast_2d(phase)

    if fftshift and amplitude.shape[1] > 0:
        amplitude = np.fft.fftshift(amplitude, axes=1)
        phase = np.fft.fftshift(phase, axes=1)

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
    """Return a trailing window of `max_packets` packets.

    If `start_time` is given, packets before that relative time are dropped
    first, then the trailing `max_packets` are kept.
    """
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
