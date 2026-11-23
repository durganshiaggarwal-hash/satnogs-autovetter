"""What the satellite actually sent: the data frames a station decoded, taken apart.

When a SatNOGS station hears a satellite it knows how to decode, it
saves each data packet it recovered as a small binary file (the
observation's ``demoddata``).  This module fetches them and reads them:

* :class:`DecodedFrame` -- one packet's bytes and when it was received;
* :func:`parse_ax25` -- the header of an **AX.25** packet, the amateur
  packet-radio format most small satellites use: destination and source
  callsigns (six characters, each shifted left one bit, plus an SSID
  byte), any repeaters, the control byte (0x03 = an unnumbered
  "information" frame) and the protocol byte (0xF0 = no layer 3);
* :func:`describe` -- everything a person might want to see: the
  header, readable text inside, the bytes in hex, the first bits.

The *radio* demodulation was done by the volunteer's station
(gr-satellites); this module reads the packets it produced.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .source import SatnogsError, Transport, UrllibTransport

OBSERVATION_URL = "https://network.satnogs.org/api/observations/{}/?format=json"
MAX_FRAMES = 60
MAX_FRAME_BYTES = 64 * 1024


@dataclass(frozen=True)
class DecodedFrame:
    data: bytes
    received_at: datetime | None
    name: str

    @property
    def size(self) -> int:
        return len(self.data)


@dataclass(frozen=True)
class AX25Header:
    destination: str
    source: str
    repeaters: tuple[str, ...]
    control: int
    pid: int | None
    info: bytes

    @property
    def kind(self) -> str:
        if self.control == 0x03:
            return "AX.25 UI frame (unnumbered information)"
        if self.control & 0x01 == 0:
            return "AX.25 I frame (numbered information)"
        return f"AX.25 control 0x{self.control:02x}"


_CALL = re.compile(r"^[A-Z0-9]{1,6}$")
_TIME = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})")


def _address(chunk: bytes) -> tuple[str, bool] | None:
    """One 7-byte AX.25 address: (callsign-SSID, is this the last address?)."""
    if len(chunk) != 7:
        return None
    call = "".join(chr(b >> 1) for b in chunk[:6]).rstrip()
    if not _CALL.match(call):
        return None
    ssid = (chunk[6] >> 1) & 0x0F
    return (f"{call}-{ssid}" if ssid else call), bool(chunk[6] & 0x01)


def parse_ax25(data: bytes) -> AX25Header | None:
    """The AX.25 header, or ``None`` if the bytes are not an AX.25 packet."""
    if len(data) < 15:
        return None
    destination = _address(data[0:7])
    source = _address(data[7:14])
    if destination is None or source is None:
        return None
    position, last = 14, source[1]
    repeaters = []
    while not last and len(repeaters) < 8:
        hop = _address(data[position:position + 7])
        if hop is None:
            return None
        repeaters.append(hop[0])
        position, last = position + 7, hop[1]
    if position >= len(data):
        return None
    control = data[position]
    position += 1
    pid = None
    if control == 0x03 or control & 0x01 == 0:   # UI and I frames carry a protocol byte
        if position >= len(data):
            return None
        pid = data[position]
        position += 1
    return AX25Header(destination[0], source[0], tuple(repeaters), control, pid, bytes(data[position:]))


def readable_text(data: bytes, shortest: int = 6) -> list[str]:
    """Runs of printable ASCII at least ``shortest`` long (telemetry often has some)."""
    runs = [m.group().decode("ascii") for m in re.finditer(rb"[\x20-\x7e]{%d,}" % shortest, data)]
    # Binary telemetry is full of accidental printable bytes ('>333>'); keep
    # only runs that are mostly letters and digits, which is what real text is.
    return [run for run in runs if sum(c.isalnum() or c == " " for c in run) >= 0.75 * len(run)
            and sum(c.isalpha() for c in run) >= 3]


def hex_rows(data: bytes, width: int = 16, rows: int = 8) -> list[str]:
    out = []
    for start in range(0, min(len(data), width * rows), width):
        chunk = data[start:start + width]
        text = "".join(chr(b) if 32 <= b < 127 else "·" for b in chunk)
        out.append(f"{start:04x}  {' '.join(f'{b:02x}' for b in chunk):<{width * 3}} {text}")
    return out


def bits(data: bytes, count: int = 128) -> str:
    return "".join(f"{b:08b}" for b in data[: (count + 7) // 8])[:count]


def describe(frame: DecodedFrame) -> dict:
    """Everything shown about one frame, as plain values."""
    header = parse_ax25(frame.data)
    payload = header.info if header is not None else frame.data
    all_text = all(32 <= b < 127 or b in (9, 10, 13) for b in frame.data) and frame.data
    return {
        "time": frame.received_at.strftime("%H:%M:%S") if frame.received_at else "",
        "size": frame.size,
        "kind": header.kind if header else ("plain text" if all_text else "binary (not AX.25)"),
        "source": header.source if header else "",
        "destination": header.destination if header else "",
        "repeaters": list(header.repeaters) if header else [],
        "payload_size": len(payload),
        "text": (frame.data.decode("ascii", "replace").strip() if all_text else " · ".join(readable_text(payload))),
        "hex": hex_rows(frame.data),
        "bits": bits(frame.data),
    }


def _received_at(name: str) -> datetime | None:
    match = _TIME.search(name)
    if not match:
        return None
    day, hh, mm, ss = match.groups()
    return datetime.fromisoformat(f"{day}T{hh}:{mm}:{ss}+00:00")


class FrameStore:
    """Fetches an observation's decoded frames once and keeps them on disk."""

    def __init__(self, root: str | Path, transport: Transport | None = None) -> None:
        self._root = Path(root) / "frames"
        self._transport = transport or UrllibTransport()

    def frames(self, obs_id: int) -> list[DecodedFrame]:
        folder = self._root / str(obs_id)
        index = folder / "index.json"
        if not index.exists():
            self._download(obs_id, folder)
        names = json.loads(index.read_text(encoding="utf-8"))
        return [DecodedFrame((folder / name).read_bytes(), _received_at(name), name)
                for name in names if (folder / name).exists()]

    def _download(self, obs_id: int, folder: Path) -> None:
        response = self._transport.get(OBSERVATION_URL.format(obs_id))
        if response.status != 200:
            raise SatnogsError(f"observation {obs_id}: SatNOGS answered {response.status}")
        entries = (response.json() or {}).get("demoddata") or []
        folder.mkdir(parents=True, exist_ok=True)
        names = []
        for entry in entries[:MAX_FRAMES]:
            url = entry.get("payload_demod") if isinstance(entry, dict) else None
            if not url:
                continue
            reply = self._transport.get(url)
            if reply.status != 200 or len(reply.body) > MAX_FRAME_BYTES:
                continue
            name = re.sub(r"[^A-Za-z0-9_.-]", "_", url.rsplit("/", 1)[-1])[:120] + ".bin"
            (folder / name).write_bytes(reply.body)
            names.append(name)
        (folder / "index.json").write_text(json.dumps(names), encoding="utf-8")
