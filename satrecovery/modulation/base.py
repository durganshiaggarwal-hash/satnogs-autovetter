"""Abstract base class for every modulation scheme.

DESIGN NOTE: ABSTRACTION
------------------------
A modulation scheme is really just two operations -- turn bits into
complex symbols, and turn received symbols back into bits.  Everything
else (how many points the constellation has, where they sit, what the
scheme is called) is detail that the rest of the program should not have
to know about.

:class:`Modulation` captures that idea as an *abstract base class*:

* It declares ``_build_constellation()`` as ``@abstractmethod``.  Python
  then refuses to instantiate any subclass that forgets to implement it,
  so the contract is enforced by the language rather than by a comment.
* It cannot be instantiated itself -- ``Modulation()`` raises
  ``TypeError``.  There is no such thing as "a modulation in general".
* It provides *concrete* implementations of ``map_bits`` and
  ``demap_symbols`` that work for any constellation.  Subclasses inherit
  the hard part for free and only supply their own geometry.

That last point is the Template Method pattern: the base class owns the
algorithm, the subclass fills in one step.

DESIGN NOTE: POLYMORPHISM
-------------------------
Because every scheme exposes the same interface, code that uses them
never needs an ``if`` statement::

    for scheme in [BPSK(), QPSK(), QAM16()]:
        symbols = scheme.map_bits(bits)      # same call
        recovered = scheme.demap_symbols(symbols)

The receiver classes, the GUI's constellation plot and the benchmark
runner are all written against :class:`Modulation` and work unchanged
when a new scheme is added.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.special import erfc

from ..core.exceptions import BitLengthError, ValidationError


def gray_code(index: int) -> int:
    """Return the reflected binary (Gray) code of ``index``.

    Consecutive Gray codes differ in exactly one bit.  Laying a
    constellation out in Gray order means that mistaking a point for its
    nearest neighbour -- by far the most likely error -- costs only a
    single bit instead of several.  That is why real systems use it, and
    why our bit error rates will match textbook curves.
    """
    return index ^ (index >> 1)


def q_function(x: np.ndarray | float) -> np.ndarray | float:
    """Gaussian tail probability ``Q(x) = P(X > x)`` for standard normal X."""
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2.0))


class Modulation(ABC):
    """Base class for all modulation schemes.

    Parameters
    ----------
    name:
        Short display name, e.g. ``"16QAM"``.
    bits_per_symbol:
        How many bits each transmitted symbol carries.
    """

    def __init__(self, name: str, bits_per_symbol: int) -> None:
        if bits_per_symbol < 1:
            raise ValidationError(
                "bits_per_symbol", bits_per_symbol, "must be at least 1"
            )

        self._name = str(name)
        self._bits_per_symbol = int(bits_per_symbol)

        # Template Method: the base class drives construction and
        # normalisation; the subclass supplies only the geometry.
        constellation = np.asarray(
            self._build_constellation(), dtype=np.complex128
        )
        expected = 2**self._bits_per_symbol
        if constellation.shape != (expected,):
            raise ValidationError(
                f"{name}.constellation",
                constellation.shape,
                f"expected exactly {expected} points",
            )

        # Normalise to unit average symbol energy.  Doing it here, once,
        # for every subclass guarantees that "SNR = 10 dB" means the same
        # thing whichever scheme is selected -- without which no
        # comparison between schemes would be meaningful.
        mean_energy = float(np.mean(np.abs(constellation) ** 2))
        self._constellation = constellation / np.sqrt(mean_energy)
        self._constellation.flags.writeable = False

    # ------------------------------------------------------------------
    # The one thing subclasses must provide
    # ------------------------------------------------------------------
    @abstractmethod
    def _build_constellation(self) -> np.ndarray:
        """Return the constellation points, ordered by bit pattern.

        Element ``i`` of the returned array is the symbol transmitted for
        the bit pattern whose unsigned big-endian value is ``i``.
        Scaling does not matter: the base class normalises the result.
        """

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Short display name of the scheme."""
        return self._name

    @property
    def bits_per_symbol(self) -> int:
        """Bits carried by each symbol."""
        return self._bits_per_symbol

    @property
    def order(self) -> int:
        """Number of constellation points, ``2 ** bits_per_symbol``."""
        return 2**self._bits_per_symbol

    @property
    def constellation(self) -> np.ndarray:
        """Read-only array of unit-energy constellation points."""
        return self._constellation

    @property
    def rotational_symmetry(self) -> int:
        """How many rotations map the constellation onto itself.

        Turn a QPSK constellation by 90 degrees and you get the same four
        points back, so its rotational symmetry is 4.

        This is not trivia -- it is what makes blind carrier recovery
        possible.  Raising the received signal to this power cancels the
        data out, because every constellation point lands on the same
        place, and what survives is a pure tone at the frequency error
        multiplied by this number.  Measure that tone, divide, and you
        have the frequency error without knowing a single transmitted
        bit.  :class:`~satrecovery.receivers.stages.CoarseFrequencyCorrection`
        does exactly this.

        The default of 4 suits any constellation with quadrant symmetry,
        which covers both QAM and APSK.  Phase shift keying overrides it
        with its own order.
        """
        return 4

    @property
    def blind_frequency_recoverable(self) -> bool:
        """Does the Nth-power trick actually work on this constellation?

        :attr:`rotational_symmetry` claims that raising the signal to
        that power collapses every constellation point onto one angle,
        leaving a clean tone.  This property **checks the claim** against
        the geometry instead of trusting it, by raising the points and
        measuring how tightly their angles cluster.

        The measurement is the magnitude-weighted mean resultant length
        of those angles: 1.0 when they all agree, 0 when they are spread
        evenly round the circle.  Weighting by magnitude is the crucial
        part, because raising to a high power exaggerates differences in
        amplitude enormously, and whichever points come out largest are
        the ones that set the tone.

        Measured across this project's six schemes, the answer falls into
        three clearly separated groups:

        ==========  =============  ==================================
        Scheme      Concentration  What that means
        ==========  =============  ==================================
        PSK         1.00           Every point lands on one angle
        16/64QAM    0.45 - 0.52    Outer points dominate and do align
        16APSK      0.005          No tone at all
        ==========  =============  ==================================

        QAM sits in the middle and still works in practice.  Its corner
        points are much further out than its inner ones -- after the
        fourth power the corners of 16QAM outweigh the innermost points
        by about eighty to one -- and those corners *do* share an angle,
        so a strong tone survives even though the inner points scatter.

        16APSK is the genuine failure.  Its two rings are 4-fold and
        12-fold symmetric, so the fourth power collapses the inner ring
        only, and the magnitude weighting that rescues QAM cannot help
        because the outer ring's twelve points spread themselves evenly.
        (The *twelfth* power does collapse both rings of the DVB-S2
        layout, and Phase 3's offline reference decoder uses it on clean
        recordings; at the noise levels a live receiver faces, raising
        noise to the twelfth power drowns the tone.)  The
        "frequency estimate" that comes back is the position of a noise
        peak, and acting on it spins a constellation that was standing
        still.  Measured here: estimates several kilohertz out even at
        20 dB, where everything else is accurate to a few hertz. DVB-S2
        receivers meet the same wall and use pilot-aided estimation for
        APSK instead.

        Deriving this from the geometry rather than hard-coding a flag
        per scheme means a scheme added later is classified correctly
        without anybody having to remember to think about it.
        """
        powered = self._constellation**self.rotational_symmetry
        magnitudes = np.abs(powered)
        total = float(np.sum(magnitudes))
        if total <= 0:
            return False
        unit = powered / np.maximum(magnitudes, 1e-300)
        concentration = abs(np.sum(magnitudes * unit)) / total
        return bool(concentration > 0.1)

    @property
    def minimum_distance(self) -> float:
        """Smallest distance between any two constellation points.

        A useful figure of merit: for a fixed average power, a larger
        minimum distance means a scheme tolerates more noise.
        """
        points = self._constellation
        diff = np.abs(points[:, None] - points[None, :])
        np.fill_diagonal(diff, np.inf)
        return float(np.min(diff))

    # ------------------------------------------------------------------
    # Concrete behaviour inherited by every subclass
    # ------------------------------------------------------------------
    def map_bits(self, bits: np.ndarray) -> np.ndarray:
        """Map a bit stream to complex symbols (the transmitter side)."""
        bits = np.asarray(bits, dtype=np.uint8).ravel()
        if bits.size % self._bits_per_symbol != 0:
            raise BitLengthError(int(bits.size), self._bits_per_symbol)
        if bits.size and not np.all((bits == 0) | (bits == 1)):
            raise ValidationError("bits", "<array>", "values must be 0 or 1")

        indices = self.bits_to_indices(bits)
        return self._constellation[indices]

    def demap_symbols(self, symbols: np.ndarray) -> np.ndarray:
        """Map received symbols back to bits by nearest-neighbour decision.

        This is the classical hard-decision demapper: pick the
        constellation point closest to each received symbol.  It is the
        optimal rule when the noise is additive, white and Gaussian, which
        makes it the right baseline to measure the learned demappers
        against.
        """
        indices = self.nearest_indices(symbols)
        return self.indices_to_bits(indices)

    def nearest_indices(self, symbols: np.ndarray) -> np.ndarray:
        """Index of the closest constellation point for each symbol."""
        symbols = np.asarray(symbols, dtype=np.complex128).ravel()
        if symbols.size == 0:
            return np.empty(0, dtype=np.int64)
        distances = np.abs(symbols[:, None] - self._constellation[None, :])
        return np.argmin(distances, axis=1).astype(np.int64)

    def decision_distances(self, symbols: np.ndarray) -> np.ndarray:
        """Distance from each symbol to its chosen constellation point.

        Phase 5 turns this into a confidence score: a symbol that lands
        far from every legal point is one the receiver should admit it is
        unsure about.
        """
        symbols = np.asarray(symbols, dtype=np.complex128).ravel()
        if symbols.size == 0:
            return np.empty(0, dtype=float)
        distances = np.abs(symbols[:, None] - self._constellation[None, :])
        return np.min(distances, axis=1)

    # ------------------------------------------------------------------
    # Bit <-> symbol-index conversion
    # ------------------------------------------------------------------
    def bits_to_indices(self, bits: np.ndarray) -> np.ndarray:
        """Pack groups of ``bits_per_symbol`` bits into integers (MSB first)."""
        bits = np.asarray(bits, dtype=np.uint8).ravel()
        if bits.size % self._bits_per_symbol != 0:
            raise BitLengthError(int(bits.size), self._bits_per_symbol)
        grouped = bits.reshape(-1, self._bits_per_symbol)
        weights = 1 << np.arange(self._bits_per_symbol - 1, -1, -1)
        return grouped.astype(np.int64) @ weights.astype(np.int64)

    def indices_to_bits(self, indices: np.ndarray) -> np.ndarray:
        """Unpack integers into groups of ``bits_per_symbol`` bits."""
        indices = np.asarray(indices, dtype=np.int64).ravel()
        if indices.size and (indices.min() < 0 or indices.max() >= self.order):
            raise ValidationError(
                "indices", "<array>", f"values must be in [0, {self.order})"
            )
        shifts = np.arange(self._bits_per_symbol - 1, -1, -1)
        return ((indices[:, None] >> shifts) & 1).astype(np.uint8).ravel()

    # ------------------------------------------------------------------
    # Reference curve -- overridden where a closed form is known
    # ------------------------------------------------------------------
    def theoretical_ber(self, ebn0_db: np.ndarray | float) -> np.ndarray | None:
        """Textbook bit error rate over AWGN, or ``None`` if unknown.

        Another use of polymorphism: the GUI simply asks the selected
        scheme for its theory curve.  Schemes with a known formula return
        one, :class:`~satrecovery.modulation.apsk.APSK16` returns
        ``None``, and the plot adapts without the GUI knowing why.
        """
        return None

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.order

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Modulation):
            return NotImplemented
        return type(self) is type(other) and self._name == other._name

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._name))

    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self._name!r}, "
            f"order={self.order}, bits_per_symbol={self._bits_per_symbol})"
        )
