# Presence calculators on the LG board

`backend/af8.py` and `backend/hybrid2_calc.py`, bundled by
`scripts/build_board_bundle.sh`, installed on the board at
`/var/csi/board-calc/` (2026-10-07).

```
af8 CAPTURE.bin [--json]     # the verdict: 1 person, 0 no person (empty room, robot vacuum)
h2  CAPTURE.bin [--json]     # hybrid 2's one-minute range rule, kept for comparison
```

Both exit 0 with a verdict; 2 when the capture cannot be judged (missing, not
MTK, under 30 s of usable data, a feature undefined); 3 on a NumPy older than
1.26. `--json` adds the evidence -- for `af8` the nine features, `p_person`,
per-stage milliseconds and peak memory; for `h2` the motion P90 and breathing
run.

## On the board, measured 2026-10-07

**The board's Python is 32-bit.** The kernel is aarch64, but `python3` is a
32-bit ARM build (`cpython-312-arm-linux-gnueabi`) and its NumPy 1.26.4 is
built for that ABI. An aarch64 NumPy cannot load into it, and PyPI has no
32-bit ARM NumPy 2. So the calculators run on the board's own NumPy -- the
one LG's detector needs anyway -- and the bundle is code only.

**Size.** 304 KB on disk (84 KB packed): eleven backend modules and two
entry points; 210 KB for `af8`'s eight alone. No NumPy, scipy or CSIKit is
shipped. Peak memory 85 MB per run, of ~770 MB available.

**Accuracy.** All 256 labelled environment-2 one-minute windows of the AF8
guide (125 person, 60 empty, 71 robot vacuum), each streamed to the board and
scored there at [0, 61):

| | board |
|---|---|
| judged | 256 of 256 |
| accuracy | 99.2 % (254) |
| people found | 98.4 % (123 / 125) |
| non-person rejected | 100 % (131 / 131), every vacuum included |
| verdicts differing from the server's | **0** |

Against the server (NumPy 2.5.2, x86-64) on the same windows: the motion
features agree to 1e-15, the breathing ones to 6.2e-5, p(person) to 2.4e-6 --
two NumPy builds rounding differently, never by enough to move a verdict. The two
misses are `20261005_145218` (still, p 0.40) and `20261005_155801` (walking,
p 0.26) -- the server misses the same two. These windows are in the model's
training set, so this measures that the board reproduces the server, not how
the model generalises; for that, the guide's held-out 99.2 % (environment
1→2) stands.

**Time**, one core (`OPENBLAS_NUM_THREADS=1`), the board otherwise at its
usual load average of ~18:

| stage | server (guide §8-3) | board, median of 256 |
|---|---|---|
| file decode | 150-300 | 608 |
| resample | 2.6 | 20 |
| A (motion) | 6 | 73 |
| F (breathing) | 82 | 1021 → ~850 with the DFT cache |
| model | < 0.002 | 0.1 |
| **features + model** | **~91-98** | **1116** (~11x) |
| per capture, decode included | | 1723 |
| `af8` end to end (Python start, imports, all of it) | | 2.0-2.1 s |

`h2` on the same capture takes 13 s: it runs FarSense over all 245
subcarriers where `af8` uses 31.

## Is more lightweighting needed?

Not for one verdict per one-minute capture: ~2 s of one core per minute is
~3 % of one of the four. Not for disk or memory either.

It would be for a verdict **every second** on a sliding 60 s window (the
guide's §9.8): 1.1 s of features does not fit in a second. The way there is
not a smaller model but not recomputing what has not changed -- sliding by 1 s
adds one new 10 s FarSense window and keeps the other 50, so the per-second
cost is one window's F (~20 ms on the board) plus A (~70 ms), around a tenth of
today's. That is new code and is not written.

Already done, with results unchanged to the bit: the in-band DFT matrix that
FarSense rebuilt for every window is cached (`farsense._band_dft`), −16 to
−19 % of F on the board, −20 % on the server. What remains of F on the board is
mostly `extract_patterns`'s matrix products, and the board's NumPy has no
optimised BLAS (`blas_opt_info: NOT AVAILABLE`, LAPACK from its built-in
`lapack_lite`) and no SIMD extensions enabled -- which is most of the 11x. Cutting the projection angles 200 → 8 saves ~20 %
(the guide) but changes the features and needs a retrain.

## Making the backend run there

All of it applies to the server too, and none of it changes a result there:

* `farsense.savgol` replaces `scipy.signal.savgol_filter` (`mode="interp"`) in
  NumPy -- 4e-14 relative at the windows FarSense uses.
* scipy (`presence.bandpass`, `farsense.highpass`) and CSIKit (`index.py`'s
  FeitCSI reader) are imported where they are used, not at the top.
* `mtk.py` hands `np.repeat` its counts as `intp`: a 32-bit NumPy refuses the
  int64 -> int32 cast.
* `farsense._band_dft` caches the in-band DFT rows.

## The bundle and the install

`scripts/build_board_bundle.sh` writes `dist/board-calc/` and
`dist/board-calc.tar.gz`: `backend/` (the eleven modules -- shipping the list
rather than the package is its test), `af8`, `h2`, `MANIFEST`, `SHA256SUMS`.
Each entry point puts the bundle on `PYTHONPATH` and runs `python3 -s -P`, so
the only `backend` it sees is the bundle's.

Board: `root@192.168.50.80` (no SFTP: copy with `ssh … 'cat > file'`, not
`scp`). `/var` is persistent ext4, `/tmp` a 1.1 GB tmpfs.

1. Nothing capturing: `ssh root@BOARD 'fuser /var/csi/capture.lock'` prints
   nothing; no capture cron is armed on the host.
2. `ssh root@BOARD 'rm -rf /var/csi/board-calc'`, then
   `gzip -dc dist/board-calc.tar.gz | ssh root@BOARD 'tar -xf - -C /var/csi'`.
3. `ssh root@BOARD 'cd /var/csi/board-calc && sha256sum -c -s SHA256SUMS && echo ok'`
4. The system NumPy is untouched (`python3 -c 'import numpy; print(numpy.__version__)'`
   → 1.26.4) -- nothing is installed outside `/var/csi/board-calc`.
5. Rollback: `rm -rf /var/csi/board-calc`.

## Running it on the board

```
/var/csi/csi_stream.sh 60 0.02 > /tmp/cap.bin && /var/csi/board-calc/af8 /tmp/cap.bin
```

* `0.02` is the stimulus interval of the environment-2 one-minute captures
  (~43-48 Hz achieved). Environment 1's five-minute captures ran at 0.05
  (~19 Hz) and are in the training set too; the features take fs from the
  data, but stay with 0.02.
* `csi_stream.sh` holds `/var/csi/capture.lock`, so an on-board run and a
  host-driven capture cannot overlap: the second exits 3 with "another capture
  already running". Choose one driver.
* A one-minute capture at 43 Hz is ~11 MB of tmpfs; delete it after scoring.
