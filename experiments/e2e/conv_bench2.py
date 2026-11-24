"""Board micro-benchmark 2: np.dot layouts and GELU forms (cyphy session, 2026-11-16)."""
import math
import time

import numpy as np

rng = np.random.default_rng(0)
F32 = np.float32


def run(name, f, reps=3):
    f()
    t = []
    for _ in range(reps):
        a = time.perf_counter(); r = f(); t.append(time.perf_counter() - a)
    print(f"{name:44s} {1000 * min(t):8.1f} ms", flush=True)
    return r


M, K, O = 8 * 800, 432, 48
A = rng.standard_normal((M, K)).astype(F32)
W = rng.standard_normal((O, K)).astype(F32)
WT = np.ascontiguousarray(W.T)
AT = np.ascontiguousarray(A.T)
print(f"--- GEMM {M} x {K} x {O}: {M * K * O / 1e6:.0f} M MAC")
r0 = run("dot(A, WT contiguous)", lambda: np.dot(A, WT))
r1 = run("dot(A, W.T view)", lambda: np.dot(A, W.T))
r2 = run("dot(W, AT) -> (O, M)", lambda: np.dot(W, AT).T)
r3 = run("matmul(A, WT)", lambda: np.matmul(A, WT))
r4 = run("einsum mk,ok->mo", lambda: np.einsum("mk,ok->mo", A, W))
r5 = run("dot(A, WT) float64", lambda: np.dot(A.astype(np.float64), WT.astype(np.float64)))
for nm, r in (("view", r1), ("W,AT", r2), ("matmul", r3), ("einsum", r4), ("f64", r5)):
    print(f"   max |{nm} - dot| {np.abs(r - r0).max():.2e}")

x = rng.standard_normal((8, 48, 32, 25)).astype(F32) * 2
P = (0.3275911, 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)


def g0(x):
    u = x * F32(1 / math.sqrt(2.0))
    a = np.abs(u)
    t = 1.0 / (1.0 + F32(P[0]) * a)
    poly = t * (F32(P[1]) + t * (F32(P[2]) + t * (F32(P[3]) + t * (F32(P[4]) + t * F32(P[5])))))
    erf = 1.0 - poly * np.exp(-a * a)
    return (0.5 * x * (1.0 + np.sign(u) * erf)).astype(F32)


def g1(x):
    """same arithmetic, in place, no sign(): 0.5 x (1 + s erf(|u|)) = x - 0.5 x (1 - erf(|u|)) for x >= 0 and
    0.5 x (1 - erf(|u|)) for x < 0, i.e. x * where(x >= 0, 1 - h, h), h = 0.5 poly exp(-u^2)."""
    a = np.abs(x); a *= F32(1 / math.sqrt(2.0))
    t = a * F32(P[0]); t += F32(1.0); np.reciprocal(t, out=t)
    p = t * F32(P[5]); p += F32(P[4]); p *= t; p += F32(P[3]); p *= t; p += F32(P[2]); p *= t; p += F32(P[1]); p *= t
    np.square(a, out=a); np.negative(a, out=a); np.exp(a, out=a); p *= a; p *= F32(0.5)
    return x * np.where(x >= 0, F32(1.0) - p, p)


print(f"--- GELU on {x.size / 1e6:.2f} M values")
y0 = run("g0 current", lambda: g0(x))
y1 = run("g1 in place", lambda: g1(x))
print(f"   max |g1 - g0| {np.abs(y1 - y0).max():.2e}")
