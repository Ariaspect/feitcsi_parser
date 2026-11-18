"""Board micro-benchmark: numpy conv2d formulations without BLAS (cyphy session, 2026-11-16)."""
import time

import numpy as np

rng = np.random.default_rng(0)
F32 = np.float32


def cols_cp(x, kh, kw, Ho, Wo, stride=1):
    """(n, C, H, W) padded -> (n, C*kh*kw, Ho*Wo) contiguous (current layout)."""
    n, C = x.shape[:2]
    sN, sC, sH, sW = x.strides
    c = np.lib.stride_tricks.as_strided(x, (n, C, kh, kw, Ho, Wo), (sN, sC, sH, sW, stride * sH, stride * sW))
    return c.reshape(n, C * kh * kw, Ho * Wo)


def cols_pc(x, kh, kw, Ho, Wo, stride=1):
    """(n, C, H, W) padded -> (n, Ho*Wo, C*kh*kw) contiguous."""
    n, C = x.shape[:2]
    sN, sC, sH, sW = x.strides
    c = np.lib.stride_tricks.as_strided(x, (n, Ho, Wo, C, kh, kw), (sN, stride * sH, stride * sW, sC, sH, sW))
    return np.ascontiguousarray(c).reshape(n, Ho * Wo, C * kh * kw)


def run(name, f, reps=3):
    f()
    t = []
    for _ in range(reps):
        a = time.perf_counter(); r = f(); t.append(time.perf_counter() - a)
    print(f"{name:40s} {1000 * min(t):8.1f} ms", flush=True)
    return r


for (C, O, H, W) in ((48, 48, 32, 25), (32, 48, 64, 49), (6, 32, 64, 49)):
    n = 8
    stride = 1 if (C, O) != (32, 48) else 2
    x = rng.standard_normal((n, C, H + 2, W + 2)).astype(F32)
    Wt = rng.standard_normal((O, C, 3, 3)).astype(F32)
    Ho = (H + 2 - 3) // stride + 1; Wo = (W + 2 - 3) // stride + 1
    Wm = Wt.reshape(O, -1)
    WmT = np.ascontiguousarray(Wm.T)
    print(f"--- conv {C}->{O} out {Ho}x{Wo}, batch {n}: {n * Ho * Wo * O * C * 9 / 1e6:.1f} M MAC")
    r0 = run("A matmul(Wm, cols C*k x P)", lambda: np.matmul(Wm, cols_cp(x, 3, 3, Ho, Wo, stride)))
    r1 = run("B matmul(cols P x CK, Wm.T view)", lambda: np.matmul(cols_pc(x, 3, 3, Ho, Wo, stride), Wm.T).transpose(0, 2, 1))
    r2 = run("C matmul(cols P x CK, WmT contiguous)", lambda: np.matmul(cols_pc(x, 3, 3, Ho, Wo, stride), WmT).transpose(0, 2, 1))
    r3 = run("D 2-D dot (nP x CK) @ (CK x O)", lambda: (cols_pc(x, 3, 3, Ho, Wo, stride).reshape(-1, C * 9) @ WmT).reshape(n, Ho * Wo, O).transpose(0, 2, 1))
    r4 = run("E 2-D dot (nP x CK) @ Wm.T view", lambda: (cols_pc(x, 3, 3, Ho, Wo, stride).reshape(-1, C * 9) @ Wm.T).reshape(n, Ho * Wo, O).transpose(0, 2, 1))
    r5 = run("F einsum tap loop", lambda: sum(np.einsum("oc,nchw->nohw", Wt[:, :, i, j], x[:, :, i:i + stride * (Ho - 1) + 1:stride, j:j + stride * (Wo - 1) + 1:stride]) for i in range(3) for j in range(3)).reshape(n, O, -1))
    r6 = run("G tensordot cols", lambda: np.tensordot(cols_pc(x, 3, 3, Ho, Wo, stride), WmT, axes=([2], [0])).transpose(0, 2, 1))
    run("cols_cp only", lambda: cols_cp(x, 3, 3, Ho, Wo, stride))
    run("cols_pc only", lambda: cols_pc(x, 3, 3, Ho, Wo, stride))
    for nm, r in (("B", r1), ("C", r2), ("D", r3), ("E", r4), ("F", r5), ("G", r6)):
        print(f"   max |{nm} - A| {np.abs(r - r0).max():.2e}")
