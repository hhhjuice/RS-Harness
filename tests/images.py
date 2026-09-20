import struct
import zlib
from pathlib import Path


def write_test_image(path: Path) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    width, height = 128, 64
    row = b"\x00" + bytes([255, 0, 0]) * (width // 2) + bytes([0, 0, 255]) * (width // 2)
    content = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(row * height))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(content)
    return content
