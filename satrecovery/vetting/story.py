"""One recording, told in a sentence anyone can follow.

The vetting station speaks to experts (probabilities, kHz, shares of a
track).  :func:`tell` combines the same two pieces of evidence -- what
the picture model thinks, and what the orbit physics says made the
track -- into a headline, one sentence and a confidence of one to five
dots.  Nothing new is decided here; it is the same evidence in plain words.
"""

from __future__ import annotations

from dataclasses import dataclass

from .orbit import TrackExplanation
from .session import Item, Policy, Verdict


@dataclass(frozen=True)
class Story:
    mark: str            # "heard", "not heard" or "unsure"
    headline: str
    sentence: str
    dots: int            # 1..5

    @property
    def symbol(self) -> str:
        return {"heard": "+", "not heard": "x"}.get(self.mark, "?")


def _short(name: str) -> str:
    return name.split(" (")[0]


def tell(item: Item, explanation: TrackExplanation | None, satellite: str, policy: Policy = Policy()) -> Story:
    """The plain-English verdict for one recording."""
    p = item.probability
    verdict = policy.verdict(p)
    sure = max(1, min(5, int(abs(p - 0.5) * 10 + 0.5)))   # halves round up (round() would give 4 for 95%)
    source = explanation.source if explanation is not None else None
    best = explanation.matches[0] if explanation is not None and explanation.matches else None
    physics_dots = 5 if best is not None and best.share >= 0.8 else 4 if best is not None and best.share >= 0.5 else 3

    if source == "ground":
        return Story("not heard", f"That wasn't {satellite}",
                     "The bright line is a radio transmitter on the ground near the station, not a satellite: "
                     "its shape is the mirror image of how the satellite moved.", physics_dots)
    if source == "other" and best is not None:
        other = _short(best.curve.name)
        return Story("not heard", f"That was {other}, not {satellite}",
                     f"{other} flew over at the same time on a nearby frequency, and its predicted path "
                     f"matches the signal in the recording.", physics_dots)
    if source == "timing":
        return Story("heard", f"Heard {satellite}, a little off schedule",
                     "Its signal is there, but drifting: the satellite was slightly ahead of or behind where "
                     "its orbit data said it would be.", physics_dots)
    if source == "scheduled" and verdict is not Verdict.NO_SIGNAL:
        return Story("heard", f"The station heard {satellite}",
                     "Its signal sits exactly where the satellite's orbit says it should.", max(sure, physics_dots))
    if verdict is Verdict.SIGNAL:
        return Story("heard", f"The station heard {satellite}",
                     f"The AI is {p:.0%} sure the satellite's signal is in this recording.", sure)
    if verdict is Verdict.NO_SIGNAL and source in (None, "no track", "unknown"):
        return Story("not heard", f"Nothing heard from {satellite}",
                     "The satellite flew over, but the recording is only noise.", sure)
    reason = ("a signal is there, but no known satellite or transmitter explains it"
              if source == "unknown" else "the evidence points both ways")
    return Story("unsure", "Not sure: a person should look", f"The AI can't decide, because {reason}.", 1)
