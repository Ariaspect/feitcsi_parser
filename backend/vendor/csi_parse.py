# Copyright (c) 2026 LG Electronics Inc. All rights reserved.
"""csi_parse.py - MediaTek MT7921 CSI parser (5 GHz, QoS Data trigger).

Turns /proc/net/wlan/csi_data dumps into model-ready arrays.

    import csi_parse as cp
    T = cp.load('csi.bin')

T holds, with P packets / Nr=2 rx / Nc=1..2 streams / K subcarriers:

    t            (P,)              seconds, relative to first packet. NOT uniform.
    H            (P, Nr, Nc, K)    complex64 channel frequency response
    rssi         (P, Nr)           int16 dBm   - required for AGC correction
    snr          (P, Nr)           int16 dB    - outlier gate, drop if > 70
    tx_valid     (P, Nc)           bool        - Nc varies per packet (AP's choice)
    active_bins  (A,)              int         - usable subcarriers, 242 of 256
    seq          (P,)              int64       - chip packet counter

`csitool.py sampletest --model` prints one packet at a time (index i,
subcarriers k taken from active_bins); its keys map here:

    key             variable                     meaning
    t               T['t'][i]                    packet time [s]
    active_bins     T['active_bins']             usable subcarrier indices
    rssi            T['rssi'][i]                 per-rx signal strength [dBm]
    snr             T['snr'][i]                  per-rx SNR [dB]
    V[k,Nr0]        T['H'][i, 0, 0, :]           CFR, rx0 / stream 0
    V[k,Nr1]        T['H'][i, 1, 0, :]           CFR, rx1 / stream 0
    CSI_ratio[k]    feature_conj(T['H'])[i,0,0]  phase-corrected signal
    check           -                            parse self-check, not model input

V is the proposal's V tensor (P, K, Nr, Nc); H is the same data with axes
(P, Nr, Nc, K), so transpose with (0, 3, 1, 2) to match.

Amplitude of H drives motion detection; phase must go through feature_conj()
first (see there). Scope: MT7921, 5 GHz, verified on 80 MHz HE-SU captures.
"""
import struct
import numpy as np

__version__ = "3.0.0"

MAGIC = 0xAC
T_VER, T_TS, T_RSSI, T_SNR, T_BW, T_MAC = 0, 2, 3, 4, 5, 7
T_I, T_Q, T_EXTRA, T_TX, T_RX, T_MODE, T_SEQ = 8, 9, 10, 15, 16, 17, 18

BW_MHZ = {0: 20, 1: 40, 2: 80}
FRAME_MODE = {1: "L-OFDM", 2: "M-HT", 3: "G-HT", 4: "VHT",
              8: "HE-SU", 11: "HE-MU", 15: "EHT"}
SNR_PLAUSIBLE = (0, 70)
ACTIVE_MIN_FRAC = 0.9
SEQ_SHIFT, SEQ_LAST_BIT, SEQ_CHAIN_MASK = 16, 1 << 15, 0x7FFF


def u(b, default=0):
    return int.from_bytes(b, "little") if b else default


def s8(b, default=0):
    return (b[0] - 256 if b[0] & 0x80 else b[0]) if b else default


def mac(b):
    return ":".join("%02x" % x for x in b) if b and len(b) == 6 else "?"


def iq(b):
    """Unpack signed 14-bit I/Q words. MT7921 sign-extends to 16 bits."""
    a = np.frombuffer(b, "<u2").astype(np.int32) & 0x3FFF
    return np.where(a & 0x2000, a - 0x4000, a)


def read_records(buf):
    """Split the TLV stream. Returns (records, stats).

    Layout: [magic 0xAC][len u16][TLV...]  with TLV = [tag u8][len u16][value].
    Next record starts at offset + 3 + len; the vendor sample code uses
    offset + len and drifts by 3 bytes per record.

    stats carries consumed/resync/trailing byte counts (they must sum to the
    file size), plus per-record offsets and TLV issues for diagnostics.
    """
    recs, offs, issues = [], [], []
    st = {"file": len(buf), "consumed": 0, "resync": 0, "trailing": 0}
    off = 0
    while off + 3 <= len(buf):
        if buf[off] != MAGIC:
            st["resync"] += 1
            off += 1
            continue
        ln = buf[off + 1] | (buf[off + 2] << 8)
        end = off + 3 + ln
        if ln == 0:
            st["resync"] += 1
            off += 1
            continue
        if end > len(buf):
            st["trailing"] = len(buf) - off
            break
        r, bad, p = {}, [], off + 3
        while p + 3 <= end:
            t = buf[p]
            tl = buf[p + 1] | (buf[p + 2] << 8)
            if p + 3 + tl > end:
                bad.append("tag %d overruns record" % t)
                break
            if t in r:
                bad.append("tag %d duplicated" % t)
            r[t] = bytes(buf[p + 3:p + 3 + tl])
            p += 3 + tl
        if p != end and not bad:
            bad.append("%d trailing bytes in record" % (end - p))
        recs.append(r)
        offs.append((off, ln))
        issues.append(bad)
        st["consumed"] += 3 + ln
        off = end
    else:
        st["trailing"] = len(buf) - off
    st["offsets"], st["issues"] = offs, issues
    return recs, st


def group_packets(recs):
    """Group per-chain records into packets.

    A new packet starts when a (tx, rx) pair repeats. Timestamps must not be
    used: MT7921 stamps chains of one packet with different milliseconds and
    grouping by them splits 55% of packets. The adjacency rule matched the
    chip's own sequence field chain-for-chain on every capture tested, and
    needs no vendor-specific field.
    """
    groups, cur, seen = [], {}, set()
    for r in recs:
        k = (u(r.get(T_TX)), u(r.get(T_RX)))
        if k in seen:
            groups.append(cur)
            cur, seen = {}, set()
        seen.add(k)
        cur[k] = r
    if cur:
        groups.append(cur)
    return groups


def occupancy(H, tx_valid=None):
    """Fraction of samples where each bin is non-zero.

    Empty Tx slots of 1-stream packets must be excluded or every active bin
    looks intermittent.
    """
    mg = np.abs(H)
    if tx_valid is None:
        return (mg > 0).reshape(-1, mg.shape[-1]).mean(0)
    m = np.broadcast_to(tx_valid[:, None, :, None], mg.shape)
    return (((mg > 0) & m).reshape(-1, mg.shape[-1]).sum(0)
            / np.maximum(m.reshape(-1, mg.shape[-1]).sum(0), 1))


def active_bins(H, tx_valid=None, min_frac=ACTIVE_MIN_FRAC):
    """Subcarrier bins that actually carry data.

    Derived from the capture rather than a tone plan: the vendor's reorder
    mapping is undocumented and its null set matches no 802.11 tone plan.
    """
    return np.where(occupancy(H, tx_valid) >= min_frac)[0]


def build_tensor(groups, peer="auto", nsub=None, require_full=True):
    """Packets -> model arrays. See module docstring for the output spec.

    peer         keep only this transmitter ("auto" = most frequent = the AP).
                 The CSI engine also latches frames not addressed to us.
    require_full keep only packets carrying every rx path. Tx streams are the
                 AP's per-packet choice and are never a completeness criterion.
    """
    if peer == "auto":
        h = {}
        for g in groups:
            for r in g.values():
                m = mac(r.get(T_MAC))
                h[m] = h.get(m, 0) + 1
        peer = max(h, key=h.get) if h else None
    if peer:
        groups = [g for g in groups
                  if all(mac(r.get(T_MAC)) == peer for r in g.values())]
    groups = [g for g in groups
              if len({len(r.get(T_I, b"")) // 2 for r in g.values()}) == 1]
    if nsub is None:
        h = {}
        for g in groups:
            k = len(next(iter(g.values())).get(T_I, b"")) // 2
            h[k] = h.get(k, 0) + 1
        nsub = max(h, key=h.get) if h else 0
    sel = [g for g in groups
           if len(next(iter(g.values())).get(T_I, b"")) // 2 == nsub]
    if not sel:
        raise ValueError("no packets match the selected peer/bandwidth")

    ntx = max(k[0] for g in sel for k in g) + 1
    nrx = max(len({k[1] for k in g}) for g in sel)
    if require_full:
        sel = [g for g in sel if len({k[1] for k in g}) == nrx]

    P = len(sel)
    H = np.zeros((P, nrx, ntx, nsub), np.complex64)
    txv = np.zeros((P, ntx), bool)
    t = np.zeros(P)
    rssi = np.zeros((P, nrx), np.int16)
    snr = np.zeros((P, nrx), np.int16)
    seq = np.zeros(P, np.int64)
    for i, g in enumerate(sel):
        head = next(iter(g.values()))
        t[i] = u(head.get(T_TS)) / 1000.0
        seq[i] = u(head.get(T_SEQ)) >> SEQ_SHIFT
        for (tx, rx), r in g.items():
            H[i, rx, tx] = iq(r[T_I]) + 1j * iq(r[T_Q])
            txv[i, tx] = True
            rssi[i, rx] = s8(r.get(T_RSSI))
            snr[i, rx] = u(r.get(T_SNR))
    head = next(iter(sel[0].values()))
    return {"t": t - t[0], "t_abs": t, "H": H, "rssi": rssi, "snr": snr,
            "seq": seq, "tx_valid": txv,
            "active_bins": active_bins(H, txv), "n_packets": P,
            "nrx": nrx, "ntx": ntx, "nsub": nsub, "peer": peer or "",
            "bw_mhz": BW_MHZ.get(u(head.get(T_BW)), 0),
            "frame_mode": u(head.get(T_MODE)), "packets": sel}


def load(path, **kw):
    """Read a capture file straight into model arrays."""
    with open(path, "rb") as f:
        recs, _ = read_records(f.read())
    return build_tensor(group_packets(recs), **kw)


def agc_scale(H, rssi, tx_valid=None, active=None):
    """Per (packet, rx) factor restoring absolute amplitude from RSSI.

    The chip strips AGC gain, so |H| is relative only; MT7921 exposes no BFAGC
    field. Pass both masks: empty Tx slots skew the RMS by 3 dB and null tones
    by 0.25 dB. This scaling is our definition, not a vendor-specified one.
    """
    A = np.abs(H) ** 2
    w = np.ones(A.shape, bool)
    if active is not None:
        sm = np.zeros(A.shape[-1], bool)
        sm[active] = True
        w &= sm[None, None, None, :]
    if tx_valid is not None:
        w &= tx_valid[:, None, :, None]
    rms = np.sqrt(np.where(w, A, 0).sum((2, 3)) / np.maximum(w.sum((2, 3)), 1))
    return 10.0 ** (rssi.astype(float) / 20.0) / np.where(rms > 0, rms, 1.0)


def apply_agc(H, rssi, tx_valid=None, active=None):
    return (H * agc_scale(H, rssi, tx_valid, active)[:, :, None, None]
            ).astype(np.complex64)


def feature_conj(H, ref_rx=-1):
    """Conjugate product across rx paths - the usable phase signal.

    Raw CSI phase carries per-frame STO/SFO/CFO error and is unusable
    (measured lag-1 coherence 0.013). Both rx chains of one packet share the
    oscillator, so the product cancels that error (coherence 0.795).
    """
    nrx = H.shape[1]
    if nrx < 2:
        raise ValueError("need >= 2 rx paths, got %d" % nrx)
    ref = nrx - 1 if ref_rx < 0 else ref_rx
    idx = [i for i in range(nrx) if i != ref]
    return (H[:, idx] * np.conj(H[:, ref:ref + 1])).astype(np.complex64)


def feature_ratio(H, ref_rx=-1):
    """Same phase as feature_conj (ratio = conj / |H_ref|^2), NaN where the
    reference is zero. Its magnitude is heavy-tailed; prefer conj."""
    nrx = H.shape[1]
    if nrx < 2:
        raise ValueError("need >= 2 rx paths, got %d" % nrx)
    ref = nrx - 1 if ref_rx < 0 else ref_rx
    idx = [i for i in range(nrx) if i != ref]
    den = H[:, ref:ref + 1]
    out = np.full((H.shape[0], len(idx)) + H.shape[2:], np.nan + 0j, np.complex64)
    np.divide(H[:, idx], den, out=out, where=np.broadcast_to(np.abs(den) > 0, out.shape))
    return out


def lag1_phase_coherence(X):
    """Circular concentration of frame-to-frame phase change. 0 = noise."""
    Z = X.reshape(X.shape[0], -1)
    a, b = Z[:-1], Z[1:]
    m = np.isfinite(a) & np.isfinite(b) & (np.abs(a) > 0) & (np.abs(b) > 0)
    if not m.any():
        return 0.0
    d = (b * np.conj(a))[m]
    return float(np.abs((d / np.abs(d)).mean()))


def export_npz(T, out, feature="all"):
    """Save model arrays. 'packets' is dropped; it holds raw record dicts."""
    act = T["active_bins"]
    d = {k: v for k, v in T.items() if k != "packets"}
    d["H_act"] = T["H"][..., act]
    d["sub_k_provisional"] = act.astype(np.int32) - T["H"].shape[-1] // 2
    d["tool_version"] = __version__
    if T["nrx"] >= 2 and feature in ("all", "conj"):
        d["conj"] = feature_conj(T["H"])[..., act]
    if T["nrx"] >= 2 and feature in ("all", "ratio"):
        d["ratio"] = feature_ratio(T["H"])[..., act]
    if feature in ("all", "agc"):
        d["H_agc"] = apply_agc(T["H"], T["rssi"], T["tx_valid"], act)[..., act]
    np.savez_compressed(out, **d)
    return d


def load_npz(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}
