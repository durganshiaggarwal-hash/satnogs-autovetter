"""Amplitude and Phase Shift Keying: 16APSK.

APSK arranges its points on concentric rings rather than a square grid.
That makes it the scheme of choice for satellite links: a ring layout has
a lower peak-to-average power ratio than square QAM of the same order, so
the satellite's amplifier can be driven harder before it starts
distorting the signal.  DVB-S2 and the newer CCSDS recommendations both
use APSK for this reason, and RML24 includes 16APSK and 32APSK.

DESIGN NOTE: WHEN A SUBCLASS CHANGES THE ANSWER
-----------------------------------------------
:class:`APSK16` overrides :meth:`Modulation.theoretical_ber` to return
``None``, because no simple closed-form expression exists for a ring
constellation with this bit mapping.  The GUI does not special-case it --
it asks every scheme for a theory curve and draws whatever it gets back.
A ``None`` simply means no reference line appears on the plot.

That is polymorphism doing real work: a subclass is allowed to answer
"not applicable", and callers written against the base class cope,
because the base class documented that possibility up front.
"""

from __future__ import annotations

import numpy as np

from .base import Modulation, gray_code


class APSK16(Modulation):
    """16APSK: 4 points on an inner ring, 12 on an outer ring.

    Parameters
    ----------
    radius_ratio:
        Ratio of the outer ring radius to the inner ring radius.  The
        default of 2.85 is the DVB-S2 value for the mid code rates and is
        close to optimal for a 4+12 layout.
    """

    #: Points on the inner ring.
    INNER_POINTS = 4
    #: Points on the outer ring.
    OUTER_POINTS = 12

    def __init__(self, radius_ratio: float = 2.85) -> None:
        self._radius_ratio = float(radius_ratio)
        super().__init__("16APSK", bits_per_symbol=4)

    @property
    def radius_ratio(self) -> float:
        """Outer-to-inner ring radius ratio (``gamma`` in the literature)."""
        return self._radius_ratio

    @property
    def papr_db(self) -> float:
        """Peak-to-average power ratio in dB.

        The number that justifies APSK's existence.  Compare it with
        16QAM's 2.55 dB: the lower the figure, the harder the satellite's
        amplifier can be driven.
        """
        peak = float(np.max(np.abs(self._constellation) ** 2))
        return 10.0 * np.log10(peak)  # average power is 1 after normalising

    def _build_constellation(self) -> np.ndarray:
        """Lay the two rings out and label them in Gray order.

        The 16 points are ordered inner-ring-first, then outer ring by
        increasing angle.  Point number ``p`` in that sequence is labelled
        ``gray_code(p)``, so stepping to the next point along a ring
        changes exactly one bit.

        This is a Gray *ordering* along the ring sequence, not the
        hand-optimised DVB-S2 mapping.  It protects the angular
        neighbours, which are the likely confusions; the inner-to-outer
        transitions are not optimised.  Since every receiver in the
        project shares this mapping, the comparison between them stays
        fair -- only the absolute bit error rate is slightly pessimistic.
        """
        ratio = self._radius_ratio

        # Normalise so the average symbol energy is 1 before the base
        # class rescales (which then becomes a no-op).  Solving
        # (4*r1^2 + 12*(ratio*r1)^2) / 16 = 1 for r1:
        inner_radius = np.sqrt(
            16.0 / (self.INNER_POINTS + self.OUTER_POINTS * ratio**2)
        )
        outer_radius = ratio * inner_radius

        sequence: list[complex] = []

        # Inner ring: 4 points at 45, 135, 225, 315 degrees.
        for k in range(self.INNER_POINTS):
            angle = np.pi / 4.0 + k * np.pi / 2.0
            sequence.append(inner_radius * np.exp(1j * angle))

        # Outer ring: 12 points every 30 degrees, starting at 15, so
        # that every inner point lines up with an outer one (45 = 15 + 30).
        # This is the DVB-S2 layout.  It used to start at 0, which put
        # the outer ring 15 degrees out of line with the standard -- a
        # mistake nothing in the simulator could reveal, because the
        # simulator and the receivers shared it.  The real RML24
        # recordings revealed it in Phase 3: decoded against the old
        # layout, their outer ring sat exactly between our points.
        for k in range(self.OUTER_POINTS):
            angle = np.pi / 12.0 + k * np.pi / 6.0
            sequence.append(outer_radius * np.exp(1j * angle))

        points = np.zeros(self.order, dtype=np.complex128)
        for position, value in enumerate(sequence):
            points[gray_code(position)] = value
        return points

    def theoretical_ber(self, ebn0_db):
        """No usable closed form for this layout -- see the class docstring."""
        return None
