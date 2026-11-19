"""Quadrature Amplitude Modulation: 16QAM and 64QAM.

QAM puts its points on a square grid, carrying information in *both*
amplitude and phase.  That buys throughput -- 64QAM moves six bits per
symbol where BPSK moves one -- at the cost of needing a much cleaner
link, since the points sit closer together for the same average power.

DESIGN NOTE: INHERITANCE AS SHARED IMPLEMENTATION, NOT SHARED LABELS
--------------------------------------------------------------------
:class:`QuadratureAmplitudeModulation` sits beside
:class:`~satrecovery.modulation.psk.PhaseShiftKeying` in the hierarchy,
not inside it.  Both are ``Modulation`` subclasses, but a QAM
constellation is a grid and a PSK constellation is a circle, so there is
genuinely nothing to reuse between them.  Inheritance is for sharing
*implementation*; when two things merely sound related, forcing one to
inherit from the other produces the classic fragile hierarchy.
"""

from __future__ import annotations

import numpy as np

from ..core.exceptions import ValidationError
from .base import Modulation, gray_code, q_function


class QuadratureAmplitudeModulation(Modulation):
    """Square QAM with Gray coding applied independently on each axis."""

    def __init__(self, order: int, name: str | None = None) -> None:
        bits_per_symbol = int(order).bit_length() - 1
        if order < 4 or (order & (order - 1)) != 0 or bits_per_symbol % 2 != 0:
            raise ValidationError(
                "order",
                order,
                "must be a power of two with an even number of bits "
                "(4, 16, 64, 256, ...)",
            )

        self._order_requested = int(order)
        self._bits_per_axis = bits_per_symbol // 2
        super().__init__(name or f"{order}QAM", bits_per_symbol)

    @property
    def bits_per_axis(self) -> int:
        """Bits carried by the I axis (the Q axis carries the same number)."""
        return self._bits_per_axis

    def _build_constellation(self) -> np.ndarray:
        """Build the grid, Gray coding each axis separately.

        The upper half of each symbol's bits selects the I level and the
        lower half selects the Q level.  Because the two axes are
        independent, a square QAM demodulator can decide them separately
        -- which is what makes QAM cheap to implement in hardware.
        """
        levels_per_axis = 1 << self._bits_per_axis

        # Amplitudes -3, -1, +1, +3 for 4 levels; -7 ... +7 for 8 levels.
        amplitudes = np.arange(levels_per_axis) * 2 - (levels_per_axis - 1)

        # Position p along the axis is labelled gray_code(p).
        level_for_label = np.zeros(levels_per_axis, dtype=float)
        for position in range(levels_per_axis):
            level_for_label[gray_code(position)] = amplitudes[position]

        points = np.zeros(self._order_requested, dtype=np.complex128)
        mask = levels_per_axis - 1
        for index in range(self._order_requested):
            i_label = index >> self._bits_per_axis
            q_label = index & mask
            points[index] = (
                level_for_label[i_label] + 1j * level_for_label[q_label]
            )
        return points

    def theoretical_ber(self, ebn0_db):
        """Standard square-QAM approximation over AWGN."""
        ebn0 = 10.0 ** (np.asarray(ebn0_db, dtype=float) / 10.0)
        m = self.order
        k = self._bits_per_symbol
        scale = (4.0 / k) * (1.0 - 1.0 / np.sqrt(m))
        argument = np.sqrt(3.0 * k * ebn0 / (m - 1.0))
        return scale * q_function(argument)


class QAM16(QuadratureAmplitudeModulation):
    """16QAM: 4 bits per symbol on a 4x4 grid.

    The common choice for satellite telemetry downlinks when the link is
    healthy -- four times the throughput of BPSK for about 7 dB more
    signal-to-noise ratio.
    """

    def __init__(self) -> None:
        super().__init__(order=16, name="16QAM")


class QAM64(QuadratureAmplitudeModulation):
    """64QAM: 6 bits per symbol on an 8x8 grid.

    Our most fragile scheme, and so the most interesting test case: it is
    where a classical receiver degrades fastest and where a learned
    demapper has the most room to help.
    """

    def __init__(self) -> None:
        super().__init__(order=64, name="64QAM")
