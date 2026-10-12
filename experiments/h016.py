"""H016 (control): lossless compressibility of bf16 weights via exponent entropy."""
import numpy as np, zlib, lzma
from common import save


def bf16_bits(x):
    u = x.astype(np.float32).view(np.uint32) >> 16          # truncate to bf16
    return u.astype(np.uint16)


def entropy(v):
    _, c = np.unique(v, return_counts=True); p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    lines, data = [], {}
    for name, w in {"нормальные N(0, 0.02)": rng.normal(0, 0.02, 4_000_000),
                    "Стьюдент t(4)·0.01 (тяжёлые хвосты)": rng.standard_t(4, 4_000_000) * 0.01}.items():
        b = bf16_bits(w)
        sign = b >> 15; exp = (b >> 7) & 0xFF; man = b & 0x7F
        hs, he, hm = entropy(sign), entropy(exp), entropy(man)
        ideal = (hs + he + hm) / 16
        raw = b.tobytes()
        # byte-plane split: exponent-heavy high byte vs mantissa-heavy low byte, then LZMA
        planes = (b >> 8).astype(np.uint8).tobytes() + (b & 0xFF).astype(np.uint8).tobytes()
        lz = len(lzma.compress(planes, preset=6)) / len(raw)
        zl = len(zlib.compress(raw, 9)) / len(raw)
        data[name] = dict(H_sign=hs, H_exp=he, H_man=hm, ideal=ideal, lzma_planes=lz, zlib=zl)
        lines.append(f"{name}: энтропия знака {hs:.2f} бит, экспоненты {he:.2f} бит (из 8), мантиссы {hm:.2f} бит (из 7)")
        lines.append(f"   теоретический предел: {100*ideal:.1f}% исходного размера; zlib {100*zl:.1f}%; LZMA по байтовым плоскостям {100*lz:.1f}%")
    save("h016", "\n".join(lines) + "\n", data)
