# SPDX-License-Identifier: Apache-2.0
"""Independent single-file TL2 conversion and rejection checks (stdlib only)."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def check(converter):
    with tempfile.TemporaryDirectory(prefix="nntr-tl2-model-") as tmp:
        root = Path(tmp)
        k, n = 96, 128
        packed = bytearray(k * n // 4)
        sums = [0] * n
        for row in range(k):
            for col in range(n):
                q = (row * 7 + col * 11) % 3 - 1
                r = row % 32
                at = ((row // 32) * (n // 32) + col // 32) * 256
                at += (r // 16) * 128 + (col % 32) * 4 + r % 4
                packed[at] |= (q + 1) << (2 * ((r // 4) % 4))
                sums[col] += q
        tail = struct.pack(f"<{2*n}f", *([0.03125] * n + sums))
        prefix, suffix = bytes(range(111)), b"untouched LM head and norm"
        original = prefix + packed + bytes((255, 0, 1, 1)) + tail + suffix
        source, output, catalog = root / "source", root / "output", root / "weights"
        source.write_bytes(original)
        catalog.write_text(f"{len(prefix)} {k} {n}\n")
        subprocess.run([converter, str(source), str(catalog), str(output)], check=True)
        result = output.read_bytes()
        native = ((k + 23) // 24) * ((n + 127) // 128) * 640
        assert len(result) == len(prefix) + native + len(tail) + len(suffix)
        assert result[:len(prefix)] == prefix
        assert result[len(prefix)+native:] == tail + suffix
        for row in range(k):
            for col in range(n):
                block, group = row // 24, (row % 24) // 3
                j = (col % 32) * 4 + col // 32
                at = len(prefix) + block * 640
                magnitude = (result[at + (group // 2) * 128 + j] >>
                             (4 * (group % 2))) & 15
                sign = (result[at + 512 + j] >> group) & 1
                value = -magnitude if sign else magnitude
                q = ((value + 13) // (3 ** (2 - row % 3))) % 3 - 1
                assert q == (row * 7 + col * 11) % 3 - 1
        failed = subprocess.run([converter, str(source), str(catalog), str(output)],
                                capture_output=True)
        assert failed.returncode != 0 and output.read_bytes() == result
        broken = bytearray(original)
        broken[len(prefix) + len(packed)] = 249  # -7 is not pure unit ternary.
        source.write_bytes(broken)
        rejected = root / "rejected"
        failed = subprocess.run([converter, str(source), str(catalog), str(rejected)],
                                capture_output=True)
        assert failed.returncode != 0 and not rejected.exists()
        assert source.read_bytes() == broken
        print("Unified TL2 conversion exact; non-experts preserved; invalid/existing output rejected")


if __name__ == "__main__":
    check(str(Path(sys.argv[1]).resolve()))
