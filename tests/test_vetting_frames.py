"""Tests for reading the data packets a station decoded (satrecovery.vetting.frames)."""

from __future__ import annotations

import json

from satrecovery.vetting import Response, Transport
from satrecovery.vetting.frames import DecodedFrame, FrameStore, describe, parse_ax25, readable_text


def ax25_address(call: str, ssid: int, last: bool) -> bytes:
    padded = call.ljust(6)[:6]
    return bytes(ord(c) << 1 for c in padded) + bytes([0x60 | (ssid << 1) | (1 if last else 0)])


def ax25_frame(src: str, dst: str, info: bytes, via: tuple[str, ...] = ()) -> bytes:
    parts = [ax25_address(dst, 11, False), ax25_address(src, 0, not via)]
    for i, hop in enumerate(via):
        parts.append(ax25_address(hop, 0, i == len(via) - 1))
    return b"".join(parts) + bytes([0x03, 0xF0]) + info


def test_reads_a_real_ax25_header():
    # The first bytes of a real KNACKSAT-2 packet from SatNOGS (HS0K -> HS0AK-11, UI frame).
    real = bytes.fromhex("90a6608296407690a660964040610 3f0".replace(" ", "")) + b"\x00" * 10
    header = parse_ax25(real)
    assert header is not None
    assert (header.source, header.destination) == ("HS0K", "HS0AK-11")
    assert header.control == 0x03 and header.pid == 0xF0 and header.kind.startswith("AX.25 UI")


def test_repeaters_and_payload():
    frame = ax25_frame("CQ1SAT", "APRS", b"hello from orbit", via=("WIDE1",))
    header = parse_ax25(frame)
    assert header.repeaters == ("WIDE1",) and header.info == b"hello from orbit"


def test_non_ax25_bytes_are_not_mistaken_for_it():
    assert parse_ax25(bytes(range(40))) is None
    assert parse_ax25(b"\x01\x02") is None
    assert parse_ax25(b"TechSat-V4.2 telemetry follows") is None


def test_readable_text_keeps_words_not_noise():
    assert readable_text(b"\x00\x01TechSat-V4.2\x00\xff") == ["TechSat-V4.2"]
    assert readable_text(b"\x00'>333>\x00>33S>Ha\x00") == []


def test_describe_plain_text_and_binary():
    text = describe(DecodedFrame(b"CQ CQ DE RS44 K", None, "x"))
    assert text["kind"] == "plain text" and text["text"] == "CQ CQ DE RS44 K"
    packet = describe(DecodedFrame(ax25_frame("HS0K", "HS0AK", b"\x00" * 20 + b"VERSION 2"), None, "x"))
    assert packet["source"] == "HS0K" and packet["destination"] == "HS0AK-11"
    assert "VERSION 2" in packet["text"] and len(packet["bits"]) == 128 and packet["hex"]


class FakeTransport(Transport):
    def __init__(self, pages):
        self.pages, self.requests = pages, []

    def get(self, url, timeout=60.0):
        self.requests.append(url)
        body = self.pages.get(url)
        if body is None:
            return Response(404, {}, b"")
        return Response(200, {}, body if isinstance(body, bytes) else json.dumps(body).encode())


def test_frame_store_downloads_once(tmp_path):
    url = "https://cdn.example/data_obs/data_77_2026-11-19T15-45-18"
    frame = ax25_frame("HS0K", "HS0AK", b"\x01\x02")
    transport = FakeTransport({"https://network.satnogs.org/api/observations/77/?format=json":
                               {"demoddata": [{"payload_demod": url}]}, url: frame})
    store = FrameStore(tmp_path, transport)
    frames = store.frames(77)
    assert len(frames) == 1 and frames[0].data == frame
    assert frames[0].received_at.strftime("%H:%M:%S") == "15:45:18"
    store.frames(77)
    assert len(transport.requests) == 2      # the second call reads the saved copy
