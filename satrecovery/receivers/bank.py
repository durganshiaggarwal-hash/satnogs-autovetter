"""One receiver per modulation, presented as a single receiver.

A learned demapper is trained for one modulation: the network for 16QAM
has four outputs and has never seen a 64QAM symbol.  A benchmark that
covers several modulations therefore needs a different hybrid receiver
for each -- and should not have to know that.

:class:`ReceiverBank` holds one receiver per modulation and hands each
frame to the right one.

DESIGN NOTE: COMPOSITE AND POLYMORPHISM TOGETHER
--------------------------------------------------
A bank *is* a :class:`~satrecovery.receivers.base.Receiver`.  The
benchmark calls ``bank.recover(frame)`` exactly as it calls
``classical.recover(frame)``, with no branch on type anywhere.  Inside,
the bank picks a member by the frame's modulation -- dispatch on data,
not on class -- and the member's own ``recover`` does the work.
"""

from __future__ import annotations

from ..core.exceptions import ReceiverError, ValidationError
from ..core.frame import Frame
from ..modulation.registry import ModulationRegistry
from .base import Receiver
from .result import RecoveryResult


class ReceiverBank(Receiver):
    """Routes each frame to the receiver built for its modulation.

    Parameters
    ----------
    members:
        Mapping from modulation name to the receiver for it.
    fallback:
        Used for any modulation without its own member.  ``None`` means
        such frames are refused, which is the safe default: silently
        handing a 64QAM frame to a 16QAM network would produce a result,
        and a meaningless one.
    """

    def __init__(
        self,
        members: dict[str, Receiver],
        fallback: Receiver | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name or "Bank")
        if not members:
            raise ValidationError("members", {}, "a bank needs at least one member")
        self._members: dict[str, Receiver] = {}
        for modulation, receiver in members.items():
            if not isinstance(receiver, Receiver):
                raise ValidationError(
                    "members", type(receiver).__name__, "every member must be a Receiver"
                )
            self._members[ModulationRegistry.create(modulation).name] = receiver
        if fallback is not None and not isinstance(fallback, Receiver):
            raise ValidationError(
                "fallback", type(fallback).__name__, "expected a Receiver"
            )
        self._fallback = fallback

    @property
    def modulations(self) -> list[str]:
        """Modulations with a dedicated member."""
        return list(self._members)

    def supports(self, modulation: str) -> bool:
        """Can this bank recover frames of ``modulation``?"""
        name = ModulationRegistry.create(modulation).name
        return name in self._members or self._fallback is not None

    def member(self, modulation: str) -> Receiver:
        """The receiver that would handle ``modulation``."""
        name = ModulationRegistry.create(modulation).name
        if name in self._members:
            return self._members[name]
        if self._fallback is not None:
            return self._fallback
        raise ReceiverError(
            f"{self.name!r} has no receiver for {name}; "
            f"it covers {', '.join(self._members)}"
        )

    def _recover(self, frame: Frame) -> RecoveryResult:
        return self.member(frame.modulation).recover(frame)

    def __len__(self) -> int:
        return len(self._members)

    def __repr__(self) -> str:
        return f"ReceiverBank({self.name!r}, {sorted(self._members)})"


__all__ = ["ReceiverBank"]
