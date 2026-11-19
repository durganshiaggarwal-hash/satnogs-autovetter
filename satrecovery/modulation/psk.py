"""Phase Shift Keying: BPSK, QPSK and 8PSK.

DESIGN NOTE: A THREE-LEVEL INHERITANCE CHAIN
--------------------------------------------
    Modulation  (abstract -- "something that maps bits to symbols")
        |
    PhaseShiftKeying  (shared geometry -- "points on a circle")
        |
    BPSK / QPSK / PSK8  (concrete -- "that circle, with M points")

The middle class is where the real reuse happens.  Every PSK scheme puts
its points evenly around a circle of constant radius, so
:meth:`PhaseShiftKeying._build_constellation` is written once and the
three leaf classes only state *how many* points and *what rotation*.

Phase modulation is the workhorse of satellite telemetry precisely
because every point has the same amplitude: the satellite's power
amplifier can run flat out, in saturation, without distorting the
constellation.
"""

from __future__ import annotations

import numpy as np

from ..core.exceptions import ValidationError
from .base import Modulation, gray_code, q_function


class PhaseShiftKeying(Modulation):
    """Equally spaced points on a circle -- the shared PSK geometry."""

    def __init__(
        self, order: int, phase_offset: float = 0.0, name: str | None = None
    ) -> None:
        if order < 2 or (order & (order - 1)) != 0:
            raise ValidationError(
                "order", order, "must be a power of two and at least 2"
            )

        self._order_requested = int(order)
        self._phase_offset = float(phase_offset)

        bits_per_symbol = int(order).bit_length() - 1
        super().__init__(name or f"{order}PSK", bits_per_symbol)

    @property
    def phase_offset(self) -> float:
        """Rotation of the whole constellation, in radians."""
        return self._phase_offset

    @property
    def rotational_symmetry(self) -> int:
        """An M-point circle maps onto itself M times per revolution.

        Overriding the base class's default of 4 matters for BPSK (2) and
        8PSK (8): using the wrong power would leave the data in the
        signal and the frequency estimate would be noise.
        """
        return self.order

    def _build_constellation(self) -> np.ndarray:
        """Place point ``gray_code(p)`` at angle ``2*pi*p/M``.

        Walking around the circle (``p`` increasing) walks through Gray
        codes, so neighbouring points differ in exactly one bit.
        """
        m = self._order_requested
        points = np.zeros(m, dtype=np.complex128)
        for position in range(m):
            angle = 2.0 * np.pi * position / m + self._phase_offset
            points[gray_code(position)] = np.exp(1j * angle)
        return points

    def theoretical_ber(self, ebn0_db):
        """Standard M-PSK approximation.

        For M = 2 and M = 4 the expression below is exact; for larger M it
        is the usual nearest-neighbour approximation, which is tight at
        the signal-to-noise ratios we care about.
        """
        ebn0 = 10.0 ** (np.asarray(ebn0_db, dtype=float) / 10.0)
        k = self._bits_per_symbol
        if self.order in (2, 4):
            return q_function(np.sqrt(2.0 * ebn0))
        argument = np.sqrt(2.0 * k * ebn0) * np.sin(np.pi / self.order)
        return (2.0 / k) * q_function(argument)


class BPSK(PhaseShiftKeying):
    """Binary PSK: 1 bit per symbol, two points at +1 and -1.

    The most robust scheme in the set.  Satellite command links fall back
    to it when the link budget is tight, because getting the command
    through matters far more than getting it through quickly.
    """

    def __init__(self) -> None:
        super().__init__(order=2, phase_offset=0.0, name="BPSK")


class QPSK(PhaseShiftKeying):
    """Quadrature PSK: 2 bits per symbol, four points at the diagonals.

    Rotated by 45 degrees so that the in-phase and quadrature components
    carry one bit each, independently.  That is why QPSK achieves twice
    the data rate of BPSK at the *same* bit error rate -- the single most
    quoted result in digital communications.
    """

    def __init__(self) -> None:
        super().__init__(order=4, phase_offset=np.pi / 4.0, name="QPSK")


class PSK8(PhaseShiftKeying):
    """8PSK: 3 bits per symbol.

    The points are now only 45 degrees apart, so 8PSK needs roughly 4 dB
    more signal-to-noise ratio than QPSK for the same bit error rate.
    That trade -- throughput against robustness -- is exactly what a
    cognitive radio exists to manage.
    """

    def __init__(self) -> None:
        super().__init__(order=8, phase_offset=0.0, name="8PSK")
