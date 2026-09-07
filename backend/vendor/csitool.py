#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 LG Electronics Inc. All rights reserved.
"""
csitool.py - Wi-Fi CSI 진단 도구 (CLI)
======================================

파싱 자체는 csi_parse.py 에 있다. 이 파일은 그 위에 얹은 진단 도구다.
  verify    파싱 정합성 전수 검사
  info      캡처 요약
  dump      TLV 주석 덤프
  export    정규 텐서를 .npz 로 저장
  selftest  합성 데이터로 파서 회귀 검사
  capture   온디바이스 캡처 (MediaTek)

사용법
------
    python3 csitool.py info     csi.bin
    python3 csitool.py verify   csi.bin
    python3 csitool.py verify   csi.bin --json r.json --census
    python3 csitool.py dump     csi.bin -n 3
    python3 csitool.py export   csi.bin -o out.npz
    python3 csitool.py selftest
    python3 csitool.py capture  -o csi.bin --sec 20 --ap 192.168.0.1

모델 코드는 이 파일이 아니라 csi_parse.py 를 import 한다.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from csi_parse import *                      # noqa: F401,F403
from csi_parse import np, __version__

HAVE_NUMPY = True


def _need_numpy():
    pass


# --- 진단 전용 어댑터 -------------------------------------------------------
# csi_parse 는 모델용이라 원시 레코드를 dict 로만 돌려준다. 검증/덤프는 레코드
# 오프셋·TLV 이상·태그별 접근이 필요하므로 여기서 얇게 감싼다.

MTK_TAGS = {
    0:  ("version", 1, "u"),       1:  ("reserved1", 1, "u"),
    2:  ("timestamp_ms", 8, "u"),  3:  ("rssi_dbm", 1, "s8"),
    4:  ("snr_db", 1, "u"),        5:  ("bandwidth", 1, "u"),
    6:  ("primary_ch_idx", 1, "u"), 7: ("tx_mac", 6, "mac"),
    8:  ("csi_i", None, "iq"),     9:  ("csi_q", None, "iq"),
    10: ("extra_info", 4, "hex32"), 15: ("tx_path_idx", 1, "u"),
    16: ("rx_path_idx", 1, "u"),   17: ("frame_mode", 1, "u"),
    18: ("seq_chain", 4, "seq"),   19: ("rx_rate", 1, "u"),
    22: ("band", 1, "u"),          23: ("tone_valid", 4, "hex32"),
    11: ("deprecated11", None, "raw"), 12: ("deprecated12", None, "raw"),
    13: ("deprecated13", None, "raw"), 14: ("deprecated14", None, "raw"),
    20: ("deprecated20", None, "raw"), 21: ("deprecated21", None, "raw"),
}
MANDATORY_TAGS = (T_VER, T_TS, T_RSSI, T_SNR, T_BW, T_MAC, T_I, T_Q, T_TX, T_RX)
BW_NSUB = {0: 64, 1: 128, 2: 256}
MTK_MAGIC, REC_HDR_LEN, TLV_HDR_LEN = MAGIC, 3, 3
IQ_BITS = 14
IQ_MASK, IQ_SIGN = 0x3FFF, 0x2000
IQ_MIN, IQ_MAX = -8192, 8191
u_le, mac_str = u, mac


def decode_iq(b):
    """signed 14-bit -> int list (numpy 없이도 동작)."""
    out = []
    for i in range(0, len(b) - 1, 2):
        v = (b[i] | (b[i + 1] << 8)) & 0x3FFF
        out.append(v - 0x4000 if v & 0x2000 else v)
    return out


@dataclass
class Chain:
    index: int
    offset: int
    rec_len: int
    tlvs: Dict[int, bytes]
    issues: List[str]

    def _u(self, t): return u(self.tlvs.get(t))
    @property
    def version(self): return self._u(T_VER)
    @property
    def ts_ms(self): return self._u(T_TS)
    @property
    def rssi(self): return s8(self.tlvs.get(T_RSSI))
    @property
    def snr(self): return self._u(T_SNR)
    @property
    def bw(self): return u(self.tlvs.get(T_BW), -1)
    @property
    def pri_ch(self): return self._u(6)
    @property
    def tx_mac(self): return mac(self.tlvs.get(T_MAC))
    @property
    def extra(self): return self._u(T_EXTRA)
    @property
    def tx_idx(self): return self._u(T_TX)
    @property
    def rx_idx(self): return self._u(T_RX)
    @property
    def frame_mode(self): return self._u(T_MODE)
    @property
    def nsub(self): return len(self.tlvs.get(T_I, b"")) // 2
    @property
    def has_seq(self): return T_SEQ in self.tlvs
    @property
    def seq(self): return self._u(T_SEQ) >> SEQ_SHIFT
    @property
    def seq_last(self): return bool(self._u(T_SEQ) & SEQ_LAST_BIT)
    @property
    def seq_chain(self): return self._u(T_SEQ) & SEQ_CHAIN_MASK

    def csi_int(self):
        return decode_iq(self.tlvs.get(T_I, b"")), decode_iq(self.tlvs.get(T_Q, b""))

    def csi_np(self):
        return (iq(self.tlvs[T_I]) + 1j * iq(self.tlvs[T_Q])).astype(np.complex64)


@dataclass
class Packet:
    seq: int
    ts_ms: int
    bw: int
    nsub: int
    tx_mac: str
    frame_mode: int
    chains: Dict[Tuple[int, int], Chain]

    @property
    def n_chain(self): return len(self.chains)


@dataclass
class Capture:
    backend: str
    chains: List[Chain]
    file_bytes: int = 0
    consumed: int = 0
    resync_bytes: int = 0
    trailing_bytes: int = 0
    source: str = ""

    def __len__(self): return len(self.chains)


def capture_from_bytes(buf, source=""):
    """진단용: 바이트 -> Capture(레코드 오프셋·이상 포함)."""
    recs, st = read_records(buf)
    chains = [Chain(i, st["offsets"][i][0], st["offsets"][i][1], r, st["issues"][i])
              for i, r in enumerate(recs)]
    return Capture("mtk-tlv", chains, st["file"], st["consumed"],
                   st["resync"], st["trailing"], source)


def load_capture(path):
    """진단용: 파일 -> Capture."""
    with open(path, "rb") as f:
        return capture_from_bytes(f.read(), path)


def group_chains(cap):
    """진단용: Capture -> (Packet 목록, 통계)."""
    groups, cur, seen = [], {}, set()
    for c in cap.chains:
        k = (c.tx_idx, c.rx_idx)
        if k in seen:
            groups.append(cur); cur, seen = {}, set()
        seen.add(k); cur[k] = c
    if cur:
        groups.append(cur)
    stats = {"key": "adjacent", "dup_chain": 0, "mixed_nsub": 0}
    packets = []
    for g in groups:
        if len({c.nsub for c in g.values()}) > 1:
            stats["mixed_nsub"] += 1
            continue
        h = next(iter(g.values()))
        packets.append(Packet(h.seq, h.ts_ms, h.bw, h.nsub, h.tx_mac,
                              h.frame_mode, g))
    if packets:
        nrx = max(len({k[1] for k in p.chains}) for p in packets)
        stats["nrx"] = nrx
        stats["rx_incomplete"] = sum(1 for p in packets
                                     if len({k[1] for k in p.chains}) < nrx)
        nss = {}
        for p in packets:
            n = len({k[0] for k in p.chains})
            nss[n] = nss.get(n, 0) + 1
        stats["nss_hist"] = nss
        stats["n_chain_full"] = max(p.n_chain for p in packets)
    return packets, stats


def tensor_from_packets(packets, **kw):
    """진단용 Packet 목록 -> csi_parse.build_tensor 입력 형식으로 변환."""
    groups = [{k: c.tlvs for k, c in p.chains.items()} for p in packets]
    T = build_tensor(groups, **kw)
    keep = {id(g): i for i, g in enumerate(T["packets"])}
    T["packets"] = [p for g, p in zip(groups, packets) if id(g) in keep]
    return T



# =============================================================================
# 1. 검증
# =============================================================================

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"


@dataclass
class Check:
    name: str
    status: str
    detail: str


class Verifier:
    """파싱이 제대로 되었는지 다각도로 검사한다.

    구조 검사(structure_*)는 numpy 없이 동작하고,
    수치 검사(numeric_*)는 numpy 가 있을 때만 수행한다.
    """

    def __init__(self, cap: Capture):
        self.cap = cap
        self.checks: List[Check] = []
        self.packets, self.grp = group_chains(cap)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append(Check(name, status, detail))

    # ---- 구조 ------------------------------------------------------------
    def structure(self) -> None:
        cap = self.cap
        n = len(cap.chains)
        self.add("레코드 파싱", INFO, "%d 레코드 / %d 바이트" % (n, cap.file_bytes))
        if n == 0:
            self.add("레코드 존재", FAIL, "파싱된 레코드가 0건")
            return

        acct = cap.consumed + cap.resync_bytes + cap.trailing_bytes
        st = PASS if (cap.resync_bytes == 0 and acct == cap.file_bytes) else \
             (WARN if acct == cap.file_bytes else FAIL)
        self.add("바이트 정산", st,
                 "consumed %d + resync %d + trailing %d = %d (파일 %d)"
                 % (cap.consumed, cap.resync_bytes, cap.trailing_bytes,
                    acct, cap.file_bytes))
        if cap.resync_bytes:
            self.add("magic 재동기", WARN,
                     "%d 바이트를 건너뜀. stride 오류이거나 캡처 중 손실"
                     % cap.resync_bytes)

        bad = [c for c in cap.chains if c.issues]
        self.add("TLV 정합", PASS if not bad else FAIL,
                 "이상 레코드 %d 건%s" % (len(bad),
                 "" if not bad else " (예: #%d %s)" % (bad[0].index, bad[0].issues[0])))

        # 필수 태그
        miss: Dict[int, int] = {}
        for c in cap.chains:
            for t in MANDATORY_TAGS:
                if t not in c.tlvs:
                    miss[t] = miss.get(t, 0) + 1
        self.add("필수 태그", PASS if not miss else FAIL,
                 "누락 없음" if not miss else
                 ", ".join("tag %d(%s) %d건" % (t, MTK_TAGS[t][0], v)
                           for t, v in sorted(miss.items())))

        # 태그별 길이 검사 + 미지 태그
        bad_len: List[str] = []
        unknown: Dict[int, int] = {}
        census: Dict[int, Dict[int, int]] = {}
        for c in cap.chains:
            for t, v in c.tlvs.items():
                census.setdefault(t, {})
                census[t][len(v)] = census[t].get(len(v), 0) + 1
                if t not in MTK_TAGS:
                    unknown[t] = unknown.get(t, 0) + 1
                    continue
                exp = MTK_TAGS[t][1]
                if exp is not None and len(v) != exp:
                    bad_len.append("tag %d len %d (기대 %d)" % (t, len(v), exp))
        self.add("태그 길이", PASS if not bad_len else FAIL,
                 "전부 규격 일치" if not bad_len else "; ".join(sorted(set(bad_len))[:5]))
        self.add("미지 태그", PASS if not unknown else WARN,
                 "없음" if not unknown else
                 ", ".join("tag %d x%d" % (t, v) for t, v in sorted(unknown.items())))
        self.census = census

        # I/Q 쌍 길이
        mismatch = sum(1 for c in cap.chains
                       if len(c.tlvs.get(T_I, b"")) != len(c.tlvs.get(T_Q, b""))
                       or len(c.tlvs.get(T_I, b"")) % 2)
        self.add("I/Q 길이 일치", PASS if mismatch == 0 else FAIL,
                 "불일치 %d 건" % mismatch)

        # 부반송파 수 vs 대역폭
        bad_bw = [(c.bw, c.nsub) for c in cap.chains
                  if BW_NSUB.get(c.bw) not in (None, c.nsub)]
        self.add("부반송파수 vs 대역폭", PASS if not bad_bw else WARN,
                 "일치" if not bad_bw else
                 "불일치 %d 건 (예 bw=%d nsub=%d)" % (len(bad_bw), bad_bw[0][0], bad_bw[0][1]))

        # 열거형 도메인
        bad_enum: List[str] = []
        if any(c.bw not in BW_NSUB for c in cap.chains):
            bad_enum.append("bandwidth")
        if any(c.frame_mode not in FRAME_MODE for c in cap.chains):
            bad_enum.append("frame_mode")
        self.add("열거값 도메인", PASS if not bad_enum else WARN,
                 "정상" if not bad_enum else "범위 밖: " + ", ".join(bad_enum))

        vers = {c.version for c in cap.chains}
        self.add("버전 일관성", PASS if len(vers) == 1 else WARN,
                 "version=%s" % sorted(vers))

        lo, hi = SNR_PLAUSIBLE
        bad_snr = [c for c in cap.chains if not (lo <= c.snr <= hi)]
        self.add("SNR 타당 범위", PASS if not bad_snr else WARN,
                 "전부 %d~%d dB 내" % (lo, hi) if not bad_snr else
                 "%d건이 범위 밖 (값 %s, 해당 MAC %s). "
                 "대상 외 프레임이거나 칩 미기입값일 수 있음"
                 % (len(bad_snr), sorted({c.snr for c in bad_snr})[:5],
                    sorted({c.tx_mac for c in bad_snr})[:3]))

        macs = {c.tx_mac for c in cap.chains}
        mh: Dict[str, int] = {}
        for c in cap.chains:
            mh[c.tx_mac] = mh.get(c.tx_mac, 0) + 1
        dom = max(mh.items(), key=lambda kv: kv[1])
        self.add("송신자 MAC", PASS if len(macs) == 1 else WARN,
                 ("단일 AP %s" % list(macs)[0]) if len(macs) == 1
                 else "%d개 혼재. 주 송신자 %s (%d/%d, %.0f%%), 그 외 %s "
                      "-> build_tensor 가 주 송신자만 사용"
                      % (len(macs), dom[0], dom[1], len(cap.chains),
                         100.0 * dom[1] / len(cap.chains),
                         sorted(m for m in macs if m != dom[0])[:3]))

    # ---- 의미/그룹핑 ------------------------------------------------------
    def semantics(self) -> None:
        cap = self.cap
        if not cap.chains:
            return

        # 타임스탬프
        ts = [c.ts_ms for c in cap.chains]
        mono = all(b >= a for a, b in zip(ts, ts[1:]))
        span = (ts[-1] - ts[0]) / 1000.0
        self.add("타임스탬프 단조성", PASS if mono else WARN,
                 "%s, 구간 %.1f s" % ("단조증가" if mono else "역행 존재", span))

        # 시퀀스
        if all(c.has_seq for c in cap.chains):
            seqs = [c.seq for c in cap.chains]
            uniq = sorted(set(seqs))
            gaps = sum(b - a - 1 for a, b in zip(uniq, uniq[1:]) if b - a > 1)
            smono = all(b >= a for a, b in zip(seqs, seqs[1:]))
            self.add("Tag18 시퀀스", PASS if (smono and gaps == 0) else WARN,
                     "패킷 %d개 (seq %d..%d), 결번 %d, %s"
                     % (len(uniq), uniq[0], uniq[-1], gaps,
                        "단조증가" if smono else "역행 존재"))
            # last-chain 플래그 일관성
            bad_last = 0
            per: Dict[int, List[Chain]] = {}
            for c in cap.chains:
                per.setdefault(c.seq, []).append(c)
            for s, g in per.items():
                if sum(1 for c in g if c.seq_last) != 1:
                    bad_last += 1
            self.add("Tag18 last-chain 플래그", PASS if bad_last == 0 else WARN,
                     "패킷당 정확히 1개" if bad_last == 0
                     else "%d 개 패킷에서 불일치" % bad_last)
            # Tag18 하위 chain 인덱스 == rx * (그 패킷의 NTX) + tx
            mism = 0
            for p in self.packets:
                ntx_p = len({k[0] for k in p.chains})
                for (tx, rx), c in p.chains.items():
                    if c.seq_chain != rx * ntx_p + tx:
                        mism += 1
            self.add("Tag18 chain == rx*NTX+tx", PASS if mism == 0 else WARN,
                     "일치 (NTX 는 패킷별 스트림수)" if mism == 0
                     else "%d/%d 건 불일치" % (mism, len(cap.chains)))
        else:
            self.add("Tag18 시퀀스", INFO, "Tag 18 부재 - 타임스탬프로 그룹핑")

        # chain 분포
        txs: Dict[int, int] = {}
        rxs: Dict[int, int] = {}
        for c in cap.chains:
            txs[c.tx_idx] = txs.get(c.tx_idx, 0) + 1
            rxs[c.rx_idx] = rxs.get(c.rx_idx, 0) + 1
        self.add("Tx path 분포", INFO, str(dict(sorted(txs.items()))))
        self.add("Rx path 분포",
                 PASS if len(rxs) >= 2 else WARN,
                 "%s%s" % (dict(sorted(rxs.items())),
                           "" if len(rxs) >= 2 else
                           "  <- Rx 경로가 1개뿐. 위상 오프셋 상쇄 불가"))

        # 패킷 그룹핑. 완전성 기준은 Rx 경로이며 Tx 스트림 수는 AP 재량이다.
        tot = len(self.packets)
        rxi = self.grp.get("rx_incomplete", 0)
        st = PASS if (tot and rxi / max(tot, 1) < 0.05) else WARN
        self.add("패킷 그룹핑", st,
                 "키=%s, 패킷 %d개, Rx 불완전 %d개(%.1f%%), chain중복 %d, nsub혼재 %d"
                 % (self.grp.get("key"), tot, rxi, 100.0 * rxi / max(tot, 1),
                    self.grp.get("dup_chain", 0), self.grp.get("mixed_nsub", 0)))
        nss = self.grp.get("nss_hist", {})
        self.add("Tx 공간스트림 분포", INFO,
                 ", ".join("%dSS %d패킷(%.1f%%)" % (k, v, 100.0 * v / max(tot, 1))
                           for k, v in sorted(nss.items()))
                 + "  (AP 가 패킷마다 선택. NRX=%d 는 고정)" % self.grp.get("nrx", 0))

        # 샘플레이트
        if len(self.packets) > 2:
            pts = [p.ts_ms for p in self.packets]
            d = sorted(b - a for a, b in zip(pts, pts[1:]))
            med = d[len(d) // 2]
            p95 = d[int(len(d) * 0.95)]
            dur = (pts[-1] - pts[0]) / 1000.0
            rate = (len(pts) - 1) / dur if dur > 0 else 0.0
            self.add("샘플레이트", INFO,
                     "실효 %.1f Hz (간격 median %d ms, p95 %d ms, max %d ms)"
                     % (rate, med, p95, d[-1]))

        # 간섭 지시자
        nbi = sum(1 for c in cap.chains if c.extra & 0x06)
        aci = sum(1 for c in cap.chains if c.extra & 0x08)
        vals: Dict[int, int] = {}
        for c in cap.chains:
            vals[c.extra] = vals.get(c.extra, 0) + 1
        top = sorted(vals.items(), key=lambda kv: -kv[1])[:4]
        self.add("간섭 지시자(Tag10)", INFO,
                 "NBI %d건 (%.1f%%), ACI %d건 (%.1f%%); 원시값 %s"
                 % (nbi, 100.0 * nbi / len(cap.chains),
                    aci, 100.0 * aci / len(cap.chains),
                    ", ".join("0x%08x x%d" % (a, b) for a, b in top)))

        modes: Dict[int, int] = {}
        for c in cap.chains:
            modes[c.frame_mode] = modes.get(c.frame_mode, 0) + 1
        self.add("프레임 모드", INFO,
                 ", ".join("%s x%d" % (FRAME_MODE.get(m, "?%d" % m), v)
                           for m, v in sorted(modes.items(), key=lambda kv: -kv[1])))

    # ---- 수치 ------------------------------------------------------------
    def numeric(self) -> None:
        if not HAVE_NUMPY:
            self.add("수치 검사", INFO, "numpy 없음 - 건너뜀")
            return
        if not self.packets:
            return
        try:
            T = tensor_from_packets(self.packets)
        except Exception as e:                     # noqa: BLE001
            self.add("텐서 구성", FAIL, str(e))
            return
        self.T = T
        H, rssi, snr, act = T["H"], T["rssi"], T["snr"], T["active_bins"]
        self.add("텐서 구성", PASS,
                 "H shape %s (패킷 x Rx x Tx x 부반송파), %d MHz, %s"
                 % (H.shape, T["bw_mhz"], FRAME_MODE.get(T["frame_mode"], "?")))

        # 14비트 범위
        mx = float(max(np.abs(H.real).max(), np.abs(H.imag).max()))
        self.add("I/Q 14비트 범위", PASS if mx <= IQ_MAX + 1 else FAIL,
                 "최대 |값| = %.0f (한계 %d). 초과 시 부호확장 오류 의심"
                 % (mx, IQ_MAX))

        # 널 마스크 안정성 (1SS 패킷의 빈 tx 슬롯은 분모에서 제외)
        nzf = occupancy(H, T.get("tx_valid"))
        unstable = int(((nzf > 0.02) & (nzf < 0.98)).sum())
        nulls = np.where(nzf < 0.02)[0]
        k = nulls.astype(int) - H.shape[-1] // 2
        self.add("널 톤 마스크 안정성", PASS if unstable == 0 else WARN,
                 "액티브 %d / %d bin, 불안정 bin %d개"
                 % (len(act), H.shape[-1], unstable))
        self.add("널 톤 위치", INFO,
                 "%d개  bin=%s  (k=i-%d 가정 시 %s)"
                 % (len(nulls), [int(x) for x in nulls[:20]],
                    H.shape[-1] // 2, [int(x) for x in k[:20]]))
        self.add("톤플랜 대조", WARN,
                 "MTK reorder 매핑이 문서에 없어 bin->부반송파 인덱스가 미확정. "
                 "널 집합이 802.11 HT/VHT/HE 톤플랜과 불일치하므로 "
                 "절대 주파수가 필요한 처리(ToF/CIR) 전에 벤더 확인 필요")

        # chain 균형 (안테나 실장 검증)
        p_rx = (np.abs(H) ** 2).mean(axis=(0, 2, 3))
        db = 10 * np.log10(p_rx / p_rx.max() + 1e-30)
        snr_m = snr.mean(axis=0)
        worst = float(-db.min())
        st = PASS if worst < 10 else (WARN if worst < 20 else FAIL)
        self.add("Rx chain 균형", st,
                 "경로별 상대전력 " + ", ".join("rx%d %+.1f dB" % (i, db[i])
                                             for i in range(len(db)))
                 + " / 평균 SNR " + ", ".join("rx%d %.1f dB" % (i, snr_m[i])
                                            for i in range(len(snr_m)))
                 + ("  <- 편차가 크면 한쪽 안테나 미실장 의심" if worst >= 10 else ""))

        # AGC 상관
        if H.shape[0] > 8:
            amp = np.sqrt((np.abs(H[:, :, 0, :]) ** 2).mean(axis=2))
            cs = []
            for r in range(H.shape[1]):
                a, b = amp[:, r].astype(float), rssi[:, r].astype(float)
                if a.std() > 0 and b.std() > 0:
                    cs.append(float(np.corrcoef(a, b)[0, 1]))
            self.add("RSSI - CSI진폭 상관", INFO,
                     ", ".join("rx%d r=%+.2f" % (i, v) for i, v in enumerate(cs))
                     + "  (칩이 AGC 제거 후 정규화하면 상관이 낮게 나온다)")

        # 위상 오프셋 상쇄 - 위상 특징 사용 가능 여부의 핵심 지표
        if H.shape[1] >= 2 and H.shape[0] > 4:
            Ha = H[:, :, 0, :][:, :, act]
            raw = lag1_phase_coherence(Ha[:, 0])
            rat = lag1_phase_coherence(feature_ratio(Ha[:, :, None, :])[:, 0, 0])
            cnj = lag1_phase_coherence(feature_conj(Ha[:, :, None, :])[:, 0, 0])
            best = max(rat, cnj)
            st = PASS if best > 0.3 else WARN
            self.add("위상 오프셋 상쇄", st,
                     "lag-1 위상 결맞음  raw %.3f -> ratio %.3f / conj %.3f "
                     "(0=랜덤, 1=완전안정)" % (raw, rat, cnj))
        elif H.shape[1] < 2:
            self.add("위상 오프셋 상쇄", FAIL,
                     "Rx 경로가 1개라 상쇄 불가. 정적(호흡) 위상 특징 사용 불가")

    def run(self) -> List[Check]:
        self.structure()
        self.semantics()
        self.numeric()
        return self.checks

    def summary(self) -> Dict[str, int]:
        out = {PASS: 0, WARN: 0, FAIL: 0, INFO: 0}
        for c in self.checks:
            out[c.status] = out.get(c.status, 0) + 1
        return out


# =============================================================================
# 2. 리포트 출력
# =============================================================================

_MARK = {PASS: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]", INFO: "[info]"}


def _disp_width(s: str) -> int:
    """한글/CJK 는 터미널에서 2칸을 차지한다. %-Ns 는 이를 반영하지 못한다."""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
               for ch in s)


def _pad(s: str, width: int) -> str:
    return s + " " * max(0, width - _disp_width(s))


def print_report(v: Verifier, cap: Capture) -> int:
    w = sys.stdout.write
    w("=" * 78 + "\n")
    w("csitool %s   검증 리포트\n" % __version__)
    w("  파일     : %s\n" % (cap.source or "<stdin>"))
    w("  백엔드   : %s\n" % cap.backend)
    w("=" * 78 + "\n")
    for c in v.checks:
        w("%s %s %s\n" % (_MARK.get(c.status, "[????]"), _pad(c.name, 26), c.detail))
    s = v.summary()
    w("-" * 78 + "\n")
    w("결과: PASS %d / WARN %d / FAIL %d / info %d\n"
      % (s[PASS], s[WARN], s[FAIL], s[INFO]))
    if s[FAIL]:
        w("=> 파싱에 문제가 있습니다. FAIL 항목을 먼저 해결하세요.\n")
    elif s[WARN]:
        w("=> 파싱은 정상입니다. WARN 항목은 캡처 조건/벤더 확인 대상입니다.\n")
    else:
        w("=> 파싱 정상.\n")
    w("=" * 78 + "\n")
    return 1 if s[FAIL] else 0


def print_census(v: Verifier) -> None:
    cen = getattr(v, "census", None)
    if not cen:
        return
    print("\nTLV 태그 census")
    print("  %-4s %-16s %-8s %s" % ("tag", "name", "count", "value-length dist"))
    for t in sorted(cen):
        name = MTK_TAGS.get(t, ("<unknown>",))[0]
        tot = sum(cen[t].values())
        print("  %-4d %-16s %-8d %s" % (t, name, tot, dict(sorted(cen[t].items()))))


# =============================================================================
# 3. 주석 헥스덤프
# =============================================================================

def _fmt_value(tag: int, v: bytes) -> str:
    kind = MTK_TAGS.get(tag, ("?", None, "raw"))[2]
    if kind == "u":
        n = u_le(v)
        if tag == T_BW:
            return "%d (DBW%d, %d부반송파)" % (n, BW_MHZ.get(n, 0), BW_NSUB.get(n, 0))
        if tag == T_MODE:
            return "%d (%s)" % (n, FRAME_MODE.get(n, "?"))
        return str(n)
    if kind == "s8":
        return "%d dBm" % s8(v)
    if kind == "mac":
        return mac_str(v)
    if kind == "hex32":
        n = u_le(v)
        if tag == T_EXTRA:
            f = []
            if n & 0x06:
                f.append("NBI")
            if n & 0x08:
                f.append("ACI")
            return "0x%08x%s" % (n, (" [" + ",".join(f) + "]") if f else "")
        return "0x%08x" % n
    if kind == "seq":
        n = u_le(v)
        return ("0x%08x  seq=%d last=%d chain=%d"
                % (n, n >> SEQ_SHIFT, 1 if n & SEQ_LAST_BIT else 0,
                   n & SEQ_CHAIN_MASK))
    if kind == "iq":
        vals = decode_iq(v)
        head = ", ".join("%d" % x for x in vals[:6])
        return "%d개 [%s, ...]" % (len(vals), head)
    return v.hex(" ")[:48]


def dump_records(cap: Capture, n: int = 3, show_hex: bool = True) -> None:
    for c in cap.chains[:n]:
        print("\n--- record #%d  @0x%06x  선언길이=%d (총 %d바이트) ---"
              % (c.index, c.offset, c.rec_len, c.rec_len + REC_HDR_LEN))
        if show_hex:
            print("  헤더: AC %02x %02x   <- magic, length LE"
                  % (c.rec_len & 0xFF, c.rec_len >> 8))
        used = REC_HDR_LEN
        for t in sorted(c.tlvs):
            v = c.tlvs[t]
            name = MTK_TAGS.get(t, ("<미지 태그>",))[0]
            print("  tag %-3d %-16s len=%-5d %s" % (t, name, len(v), _fmt_value(t, v)))
            used += TLV_HDR_LEN + len(v)
        print("  소비 %d / 레코드 %d 바이트  %s"
              % (used, c.rec_len + REC_HDR_LEN,
                 "OK" if used == c.rec_len + REC_HDR_LEN else "<-- 불일치!"))
        for m in c.issues:
            print("  !! %s" % m)


# =============================================================================
# 4. 셀프테스트 - 회귀 검사
# =============================================================================

def _enc_tlv(tag: int, val: bytes) -> bytes:
    return bytes([tag & 0xFF, len(val) & 0xFF, (len(val) >> 8) & 0xFF]) + val


def _enc_iq(vals: Sequence[int]) -> bytes:
    out = bytearray()
    for v in vals:
        out += struct.pack("<H", v & IQ_MASK)
    return bytes(out)


def _enc_record(tlvs: Sequence[Tuple[int, bytes]]) -> bytes:
    body = b"".join(_enc_tlv(t, v) for t, v in tlvs)
    return bytes([MTK_MAGIC]) + struct.pack("<H", len(body)) + body


def make_synthetic(n_pkt: int = 12, nsub: int = 64, nrx: int = 2, ntx: int = 1,
                   nulls: Sequence[int] = (0, 31, 32, 33),
                   nss_pattern: Optional[Sequence[int]] = None) -> Tuple[bytes, dict]:
    """알려진 값으로 MTK TLV 스트림을 합성한다.

    파서가 이 스트림에서 원래 값을 비트단위로 복원하는지 확인하기 위한 것.
    실장비가 없어도 파서 정확성을 증명할 수 있다.

    nss_pattern 을 주면 패킷마다 Tx 스트림 수를 다르게 만든다.
    실장비에서 AP 가 1SS/2SS 를 섞어 보내는 상황을 재현하며,
    Tag 18 chain 인덱스가 패킷별 NTX 기준임을 검증하는 데 쓴다.
    """
    truth = {"i": [], "q": [], "rssi": [], "snr": [], "ts": [],
             "seq": [], "rx": [], "tx": []}
    buf = bytearray()
    for p in range(n_pkt):
        ts = 1000000 + p * 52
        ntx_p = ntx if nss_pattern is None else nss_pattern[p % len(nss_pattern)]
        for rx in range(nrx):
            for tx in range(ntx_p):
                # 결정론적이면서 부호/포화 경계를 밟는 패턴
                I, Q = [], []
                for k in range(nsub):
                    if k in nulls:
                        I.append(0); Q.append(0); continue
                    a = ((p * 37 + k * 11 + rx * 5) % 16383) - 8192
                    b = ((p * 53 + k * 7 + rx * 3) % 16383) - 8192
                    I.append(max(IQ_MIN, min(IQ_MAX, a)))
                    Q.append(max(IQ_MIN, min(IQ_MAX, b)))
                rssi = -30 - ((p + rx) % 60)             # -30 .. -89 dBm
                snr = 20 + ((p * 3 + rx) % 30)
                last = (rx == nrx - 1 and tx == ntx_p - 1)
                chain_i = rx * ntx_p + tx
                seqw = (((p + 1) << SEQ_SHIFT)
                        | (SEQ_LAST_BIT if last else 0) | chain_i)
                bwc = {64: 0, 128: 1, 256: 2}[nsub]
                buf += _enc_record([
                    (T_VER, bytes([13])),
                    (1, bytes([3])),
                    (T_TS, struct.pack("<Q", ts)),
                    (T_RSSI, bytes([rssi & 0xFF])),
                    (T_SNR, bytes([snr])),
                    (T_BW, bytes([bwc])),
                    (6, bytes([0])),
                    (T_MAC, bytes.fromhex("3068939d5d5d")),
                    (T_I, _enc_iq(I)),
                    (T_Q, _enc_iq(Q)),
                    (T_EXTRA, struct.pack("<I", 0)),
                    (T_TX, bytes([tx])),
                    (T_RX, bytes([rx])),
                    (T_MODE, bytes([8])),
                    (T_SEQ, struct.pack("<I", seqw)),
                    (19, bytes([0])),
                ])
                truth["i"].append(I); truth["q"].append(Q)
                truth["rssi"].append(rssi); truth["snr"].append(snr)
                truth["ts"].append(ts); truth["seq"].append(p + 1)
                truth["rx"].append(rx); truth["tx"].append(tx)
    n_rec = (n_pkt * nrx * ntx if nss_pattern is None
             else sum(nrx * nss_pattern[p % len(nss_pattern)]
                      for p in range(n_pkt)))
    truth.update(n_pkt=n_pkt, nsub=nsub, nrx=nrx, ntx=ntx,
                 n_rec=n_rec, nulls=list(nulls),
                 nss_pattern=list(nss_pattern) if nss_pattern else None)
    return bytes(buf), truth


def selftest(verbose: bool = True) -> int:
    """회귀 검사. 실장비에서 찾아낸 버그가 다시 들어오는 것을 막는 것이 목적이다.

    지금까지 발견된 버그는 전부 실데이터나 탐침에서 나왔고 이 스위트가
    선제적으로 잡은 것은 없다. 즉 이것은 '발견' 도구가 아니라 '고정' 도구다.
    다만 실데이터가 밟지 않는 경계 조건(예: 상위 2비트가 채워진 I/Q 워드)은
    여기서만 확인할 수 있으므로 verify 와 상호 보완 관계다.
    """
    fails: List[str] = []
    ok: List[str] = []

    def chk(name: str, cond: bool, extra: str = "") -> None:
        (ok if cond else fails).append(name + (" :: " + extra if extra else ""))
        if verbose:
            print("%s %s%s" % (_MARK[PASS if cond else FAIL], name,
                               ("  " + extra) if extra else ""))

    # ---- 1. 라운드트립 정확도 -------------------------------------------
    print("\n[1] 합성 스트림 라운드트립")
    for nsub, nrx, ntx in ((64, 2, 1), (256, 2, 1), (128, 1, 1), (64, 2, 2)):
        buf, tr = make_synthetic(n_pkt=10, nsub=nsub, nrx=nrx, ntx=ntx)
        cap = capture_from_bytes(buf, source="<synthetic>")
        tag = "nsub=%d nrx=%d ntx=%d" % (nsub, nrx, ntx)
        chk("레코드 수 " + tag, len(cap.chains) == tr["n_rec"],
            "%d / 기대 %d" % (len(cap.chains), tr["n_rec"]))
        chk("바이트 전량 소비 " + tag,
            cap.consumed == len(buf) and cap.resync_bytes == 0
            and cap.trailing_bytes == 0,
            "consumed=%d resync=%d trailing=%d file=%d"
            % (cap.consumed, cap.resync_bytes, cap.trailing_bytes, len(buf)))
        chk("TLV 이상 없음 " + tag, all(not c.issues for c in cap.chains))

        bad = []
        for n, c in enumerate(cap.chains):
            I, Q = c.csi_int()
            if I != tr["i"][n]:
                bad.append("#%d I" % n)
            elif Q != tr["q"][n]:
                bad.append("#%d Q" % n)
            elif c.rssi != tr["rssi"][n]:
                bad.append("#%d rssi %d!=%d" % (n, c.rssi, tr["rssi"][n]))
            elif c.snr != tr["snr"][n]:
                bad.append("#%d snr" % n)
            elif c.ts_ms != tr["ts"][n]:
                bad.append("#%d ts" % n)
            elif c.seq != tr["seq"][n]:
                bad.append("#%d seq" % n)
            elif c.rx_idx != tr["rx"][n] or c.tx_idx != tr["tx"][n]:
                bad.append("#%d path" % n)
            elif c.nsub != nsub:
                bad.append("#%d nsub" % n)
        chk("전 필드 비트단위 복원 " + tag, not bad, "불일치 %s" % bad[:3])

        pk, gs = group_chains(cap)
        chk("패킷 그룹핑 " + tag,
            len(pk) == tr["n_pkt"] and all(p.n_chain == nrx * ntx for p in pk),
            "패킷 %d개, chain/패킷 %s"
            % (len(pk), sorted({p.n_chain for p in pk})))

        if HAVE_NUMPY:
            T = tensor_from_packets(pk)
            chk("텐서 shape " + tag,
                T["H"].shape == (tr["n_pkt"], nrx, ntx, nsub), str(T["H"].shape))
            exp_act = sorted(set(range(nsub)) - set(tr["nulls"]))
            chk("액티브 부반송파 유도 " + tag,
                list(T["active_bins"]) == exp_act,
                "%d개 / 기대 %d개" % (len(T["active_bins"]), len(exp_act)))
            # 첫 chain 의 I/Q 가 텐서에 그대로 들어갔는지
            chk("텐서 값 일치 " + tag,
                bool(np.all(T["H"][0, 0, 0].real == np.array(tr["i"][0], dtype=np.float32))
                     and np.all(T["H"][0, 0, 0].imag == np.array(tr["q"][0], dtype=np.float32))))

    # ---- 2. 경계값 ------------------------------------------------------
    print("\n[2] signed 14-bit 경계값")
    edge = [0, 1, -1, IQ_MAX, IQ_MIN, IQ_MAX - 1, IQ_MIN + 1, 4096, -4096]
    rec = _enc_record([
        (T_VER, bytes([13])), (T_TS, struct.pack("<Q", 1)),
        (T_RSSI, bytes([0x80])),          # -128 dBm
        (T_SNR, bytes([0])), (T_BW, bytes([0])),
        (T_MAC, b"\x00" * 6),
        (T_I, _enc_iq(edge)), (T_Q, _enc_iq(edge[::-1])),
        (T_TX, bytes([0])), (T_RX, bytes([0])),
    ])
    cap = capture_from_bytes(rec)
    I, Q = cap.chains[0].csi_int()
    chk("14비트 부호확장", I == edge and Q == edge[::-1], "%s" % I)
    chk("RSSI 부호 (0x80 -> -128)", cap.chains[0].rssi == -128,
        str(cap.chains[0].rssi))
    hi = _enc_iq([0x3FFF])                  # 상위 2비트가 채워진 워드
    chk("상위 2비트 무시", decode_iq(struct.pack("<H", 0xFFFF)) == decode_iq(hi),
        "0xFFFF -> %s" % decode_iq(struct.pack("<H", 0xFFFF)))

    # ---- 3. 고의 손상 검출 ----------------------------------------------
    print("\n[3] 고의 손상 검출")
    base, _ = make_synthetic(n_pkt=6, nsub=64)

    # 3-1 stride 오류를 유발하는 쓰레기 바이트 삽입
    mid = len(base) // 2
    inj = base[:mid] + b"\xde\xad\xbe\xef" + base[mid:]
    cap = capture_from_bytes(inj)
    chk("삽입 바이트 -> resync 보고", cap.resync_bytes > 0,
        "resync=%d" % cap.resync_bytes)

    # 3-2 레코드 길이 필드 훼손
    cor = bytearray(base)
    cor[1] = (cor[1] + 4) & 0xFF            # 첫 레코드 길이를 4 늘림
    cap = capture_from_bytes(bytes(cor))
    v = Verifier(cap); v.structure()
    detected = any(c.status == FAIL for c in v.checks) or cap.resync_bytes > 0
    chk("길이 훼손 -> 검출", detected,
        "resync=%d, FAIL=%d" % (cap.resync_bytes,
                                sum(1 for c in v.checks if c.status == FAIL)))

    # 3-3 필수 태그 누락
    partial = _enc_record([(T_VER, bytes([13])), (T_TS, struct.pack("<Q", 1))])
    v = Verifier(capture_from_bytes(partial)); v.structure()
    chk("필수 태그 누락 -> FAIL",
        any(c.name == "필수 태그" and c.status == FAIL for c in v.checks))

    # 3-4 잘린 꼬리
    cap = capture_from_bytes(base[:-20])
    chk("잘린 마지막 레코드 -> trailing 보고", cap.trailing_bytes > 0,
        "trailing=%d" % cap.trailing_bytes)

    # 3-5 Rx 경로 1개 -> 위상 상쇄 불가 판정
    if HAVE_NUMPY:
        one, _ = make_synthetic(n_pkt=8, nsub=64, nrx=1)
        v = Verifier(capture_from_bytes(one)); v.run()
        chk("Rx 1경로 -> 위상 상쇄 FAIL",
            any(c.name == "위상 오프셋 상쇄" and c.status == FAIL for c in v.checks))

    # ---- 3b. 1SS/2SS 혼재 (실장비 회귀) ---------------------------------
    print("\n[3b] Tx 스트림 수 혼재 (AP 가 1SS/2SS 를 섞어 보내는 경우)")
    buf, tr = make_synthetic(n_pkt=12, nsub=64, nrx=2, ntx=2, nss_pattern=[2, 1])
    cap = capture_from_bytes(buf)
    chk("레코드 수 (혼재)", len(cap.chains) == tr["n_rec"],
        "%d / 기대 %d" % (len(cap.chains), tr["n_rec"]))
    pk, gs = group_chains(cap)
    chk("패킷 수 (혼재)", len(pk) == tr["n_pkt"], "%d" % len(pk))
    chk("Rx 불완전 0 (1SS 를 손실로 오판하지 않음)",
        gs.get("rx_incomplete") == 0, "rx_incomplete=%s" % gs.get("rx_incomplete"))
    chk("Tx 스트림 분포 검출", gs.get("nss_hist") == {1: 6, 2: 6},
        str(gs.get("nss_hist")))
    mism = 0
    for p in pk:
        ntx_p = len({k[0] for k in p.chains})
        for (tx, rx), c in p.chains.items():
            if c.seq_chain != rx * ntx_p + tx:
                mism += 1
    chk("Tag18 chain = rx*(패킷별 NTX)+tx", mism == 0, "불일치 %d" % mism)
    mism_g = 0
    for p in pk:
        for (tx, rx), c in p.chains.items():
            if c.seq_chain != rx * 2 + tx:
                mism_g += 1
    chk("전역 NTX 규칙은 실제로 틀림 (대조군)", mism_g > 0,
        "전역 규칙 불일치 %d 건" % mism_g)
    if HAVE_NUMPY:
        T = tensor_from_packets(pk)
        chk("혼재 캡처에서 패킷 손실 없음", T["H"].shape[0] == tr["n_pkt"],
            "텐서 패킷 %d / 전체 %d" % (T["H"].shape[0], tr["n_pkt"]))
        chk("tx_valid 가 1SS 패킷을 표시",
            int(T["tx_valid"][:, 1].sum()) == 6,
            "tx1 유효 패킷 %d개" % int(T["tx_valid"][:, 1].sum()))
        exp_act = sorted(set(range(64)) - set(tr["nulls"]))
        chk("혼재 캡처에서 액티브 부반송파 정상 유도",
            list(T["active_bins"]) == exp_act,
            "%d개 / 기대 %d개" % (len(T["active_bins"]), len(exp_act)))
        occ = occupancy(T["H"], T["tx_valid"])
        chk("빈 tx 슬롯이 널 판정을 오염시키지 않음",
            int(((occ > 0.02) & (occ < 0.98)).sum()) == 0,
            "불안정 bin %d개" % int(((occ > 0.02) & (occ < 0.98)).sum()))

    # ---- 3c. 그룹핑 규칙 / 특징 안전성 -----------------------------------
    print("\n[3c] 그룹핑 규칙과 특징 안전성")
    buf, tr = make_synthetic(n_pkt=10, nsub=64, nrx=2, ntx=2, nss_pattern=[2, 1])
    cap = capture_from_bytes(buf)
    ref, _ = group_chains(cap)
    chk("인접규칙이 실제 패킷 수와 일치", len(ref) == tr["n_pkt"],
        "%d / 기대 %d" % (len(ref), tr["n_pkt"]))
    chk("인접규칙이 칩 시퀀스(Tag18)와 일치",
        all(len({c.seq for c in p.chains.values()}) == 1 for p in ref)
        and len({p.seq for p in ref}) == len(ref),
        "패킷별 seq 유일")

    # Tag 18 이 없어도(시퀀스 필드 없는 벤더) 동일하게 묶여야 한다
    cap_ns = capture_from_bytes(buf)
    for c in cap_ns.chains:
        c.tlvs.pop(T_SEQ, None)
    pk_ns, _ = group_chains(cap_ns)
    chk("Tag18 없어도 동일하게 그룹핑",
        len(pk_ns) == len(ref) and all(set(a.chains) == set(b.chains)
                                       for a, b in zip(pk_ns, ref)),
        "%d 패킷" % len(pk_ns))

    # 같은 패킷의 chain 에 서로 다른 ms 가 찍히는 실제 칩 거동
    cap_ts = capture_from_bytes(buf)
    for i, c in enumerate(cap_ts.chains):
        if i % 4 >= 2:
            c.tlvs[T_TS] = struct.pack("<Q", u(c.tlvs[T_TS]) + 1)
    by_ts = len({c.ts_ms for c in cap_ts.chains})
    pk_adj, _ = group_chains(cap_ts)
    chk("타임스탬프로 묶으면 쪼개짐 (대조군)", by_ts > len(ref),
        "고유 ts %d개 vs 실제 %d 패킷" % (by_ts, len(ref)))
    chk("인접규칙은 ms 가 흩어져도 안 쪼개짐", len(pk_adj) == len(ref),
        "%d 패킷" % len(pk_adj))

    if HAVE_NUMPY:
        T = tensor_from_packets(ref)
        r = feature_ratio(T["H"])
        act, txv = T["active_bins"], T["tx_valid"]
        nan = np.isnan(r[:, 0, :, :][:, :, act].real)
        expect = np.broadcast_to(~txv[:, :, None], nan.shape)
        chk("ratio 의 NaN 이 tx_valid=False 와 정확히 일치",
            bool(np.all(nan == expect)),
            "NaN %d / 기대 %d" % (int(nan.sum()), int(expect.sum())))
        Hd = T["H"].copy()
        Hd[:, 1] = 0
        chk("죽은 기준 경로 -> 전부 NaN (원시값 누출 없음)",
            bool(np.all(np.isnan(feature_ratio(Hd).real))))
        one, _ = make_synthetic(n_pkt=4, nsub=64, nrx=1)
        H1 = tensor_from_packets(group_chains(capture_from_bytes(one))[0])["H"]
        for nm, fn in (("ratio", feature_ratio), ("conj", feature_conj)):
            try:
                fn(H1)
                chk("NRX=1 에서 %s 거부" % nm, False, "예외 없이 통과했다")
            except ValueError:
                chk("NRX=1 에서 %s 거부" % nm, True)
        cap_rf = capture_from_bytes(buf)
        del cap_rf.chains[3]
        del cap_rf.chains[2]
        pk_rf, gs_rf = group_chains(cap_rf)
        chk("Rx 불완전 패킷 검출", gs_rf.get("rx_incomplete") == 1,
            "rx_incomplete=%s" % gs_rf.get("rx_incomplete"))
        n_t = tensor_from_packets(pk_rf, require_full=True)["H"].shape[0]
        n_f = tensor_from_packets(pk_rf, require_full=False)["H"].shape[0]
        chk("require_full 이 실제로 걸러냄", n_f > n_t,
            "True %d 패킷 / False %d 패킷" % (n_t, n_f))

    # ---- 4. 위상 상쇄 수학 검증 ------------------------------------------
    if HAVE_NUMPY:
        print("\n[4] 위상 오프셋 상쇄 수학 검증 (모의 채널)")
        rng = np.random.default_rng(20260824)
        P, S = 300, 52
        base_h = (rng.normal(size=(2, S)) + 1j * rng.normal(size=(2, S)))
        drift = rng.uniform(-np.pi, np.pi, size=P)        # 프레임별 랜덤 위상 오프셋
        Hs = np.zeros((P, 2, 1, S), dtype=np.complex64)
        for p in range(P):
            slow = np.exp(1j * 0.01 * p)                  # 느린 실제 채널 변화
            Hs[p, :, 0, :] = base_h * slow * np.exp(1j * drift[p])
        raw = lag1_phase_coherence(Hs[:, 0, 0])
        rat = lag1_phase_coherence(feature_ratio(Hs)[:, 0, 0])
        cnj = lag1_phase_coherence(feature_conj(Hs)[:, 0, 0])
        chk("공통 위상 오프셋이 raw 위상을 파괴", raw < 0.2, "raw=%.3f" % raw)
        chk("ratio 가 오프셋을 제거", rat > 0.95, "ratio=%.3f" % rat)
        chk("conj 가 오프셋을 제거", cnj > 0.95, "conj=%.3f" % cnj)

        # AGC 보정: 진폭을 임의 스케일해도 보정 후 복원되는지
        scale = rng.uniform(0.1, 10.0, size=(P, 2))
        Hscaled = (Hs * scale[:, :, None, None]).astype(np.complex64)
        rssi = np.full((P, 2), -50, dtype=np.int16)
        Hc = apply_agc(Hscaled, rssi)
        rms = np.sqrt((np.abs(Hc) ** 2).mean(axis=(2, 3)))
        chk("AGC 보정 후 RMS 일정", float(rms.std() / rms.mean()) < 1e-4,
            "변동계수 %.2e" % float(rms.std() / rms.mean()))

        # AGC 항등식: 실제 조건(1SS 혼재 + 널톤)에서 rms == 10^(RSSI/20)
        buf2, _ = make_synthetic(n_pkt=16, nsub=64, nrx=2, ntx=2,
                                 nss_pattern=[2, 1])
        T2 = tensor_from_packets(group_chains(capture_from_bytes(buf2))[0])
        a2, v2, r2 = T2["active_bins"], T2["tx_valid"], T2["rssi"]
        Hc2 = apply_agc(T2["H"], r2, v2, a2)
        w = np.zeros(Hc2.shape, dtype=bool)
        w[:, :, :, a2] = True
        w &= v2[:, None, :, None]
        got = np.sqrt((np.where(w, np.abs(Hc2) ** 2, 0).sum(axis=(2, 3))
                       / np.maximum(w.sum(axis=(2, 3)), 1)))
        want = 10.0 ** (r2 / 20.0)
        err = float(np.abs(20 * np.log10(got / want)).max())
        chk("AGC 항등식 rms==10^(RSSI/20) (1SS 혼재+널톤)", err < 1e-3,
            "최대 오차 %.2e dB" % err)
        naive = apply_agc(T2["H"], r2)          # 마스크 없이 (대조군)
        gotn = np.sqrt((np.where(w, np.abs(naive) ** 2, 0).sum(axis=(2, 3))
                        / np.maximum(w.sum(axis=(2, 3)), 1)))
        errn = float(np.abs(20 * np.log10(gotn / want)).max())
        chk("마스크 없이 보정하면 실제로 틀림 (대조군)", errn > 1.0,
            "최대 오차 %.2f dB" % errn)

    print("\n" + "-" * 78)
    print("셀프테스트: 통과 %d / 실패 %d" % (len(ok), len(fails)))
    if fails:
        for f in fails:
            print("  FAIL: %s" % f)
    print("-" * 78)
    return 1 if fails else 0


# =============================================================================
# 5. 온디바이스 캡처 (MediaTek)
# =============================================================================

CAPTURE_NOTES = """\
캡처 전 확인사항 (MT7921 실측에서 확인된 함정)
  1) iwpriv 인자에 굽은 따옴표를 쓰면 드라이버가 문자열을 인식하지 못한다.
     반드시 곧은 따옴표(')를 쓸 것.
  2) QoS Data 트리거(frame type 34)는 자기 MAC 이 수신주소인 프레임이
     실제로 와야 CSI 가 생성된다 => wlan0 이 AP 에 연결되어 있어야 한다.
  3) ping 이 유선(eth0)으로 빠지면 CSI 가 생기지 않는다.
     반드시 'ping -I wlan0' 로 인터페이스를 고정할 것.
  4) /proc/net/wlan/csi_data 읽기는 블로킹이다. 데이터가 없으면 무한 대기하므로
     timeout 으로 감쌀 것.
"""


def device_capture(out: str, seconds: int = 20, iface: str = "wlan0",
                   ap_ip: Optional[str] = None, frame_type: int = 34,
                   iwpriv: str = "iwpriv", band: int = 0,
                   ping_interval: float = 0.02) -> int:
    """타깃 보드에서 직접 캡처한다 (MediaTek 전용)."""
    def run(cmd: List[str]) -> Tuple[int, str]:
        p = subprocess.run(cmd, capture_output=True, text=True)
        return p.returncode, (p.stdout + p.stderr).strip()

    print(CAPTURE_NOTES)
    for arg in ("set_csi 2 0 %d" % band,
                "set_csi 2 3 0 %d" % frame_type,
                "set_csi 2 5 2"):
        rc, msg = run([iwpriv, iface, "driver", arg])
        print("  %s '%s' -> rc=%d %s" % (iwpriv, arg, rc, msg))
        if rc != 0:
            print("  설정 실패. iwpriv 경로와 인터페이스명을 확인하세요.")
            return 1

    # CSI 는 수신 프레임 1건당 1회 생성되므로 샘플레이트는 트래픽 속도에 종속된다.
    # 기본 ping(1 Hz)으로는 1.2 Hz 밖에 안 나와 호흡 대역 관측이 불가능하다.
    pinger = None
    if ap_ip:
        pinger = subprocess.Popen(
            ["ping", "-i", str(ping_interval), "-I", iface, ap_ip],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("  트래픽 생성: ping -i %g -I %s %s  (목표 %.0f Hz)"
              % (ping_interval, iface, ap_ip, 1.0 / max(ping_interval, 1e-6)))

    run([iwpriv, iface, "driver", "set_csi 1"])
    print("  캡처 %d초 -> %s" % (seconds, out))
    try:
        with open(out, "wb") as f:
            p = subprocess.Popen(["timeout", str(seconds), "cat",
                                  "/proc/net/wlan/csi_data"], stdout=f)
            p.wait()
    finally:
        run([iwpriv, iface, "driver", "set_csi 0"])
        if pinger:
            pinger.terminate()
    size = os.path.getsize(out) if os.path.exists(out) else 0
    print("  캡처 완료: %d 바이트" % size)
    if size == 0:
        print("  0 바이트입니다. 위 확인사항 1~4 를 점검하세요.")
        return 1
    return 0


# =============================================================================
# 6. sampletest - 파싱된 값을 key: value 로 전량 출력
# =============================================================================

def _cx(z, sig: Optional[int] = None) -> str:
    """복소수를 사람이 읽는 형식으로.

    sig 를 주면 유효숫자 표기. ratio 처럼 1 부근의 값은 정수 반올림하면
    전부 같아 보여서 검증이 안 된다.
    """
    if np.isnan(z.real) or np.isnan(z.imag):
        return "nan"
    if sig:
        return "%+.*g%+.*gj" % (sig, z.real, sig, z.imag)
    return "%+.0f%+.0fj" % (z.real, z.imag)


def _kv(k: str, v, unit: str = "", note: str = "") -> None:
    print("    %-20s %s%s%s" % (k, v, (" " + unit) if unit else "",
                                ("   " + note) if note else ""))


def _sample_model(i, p, T, Hg, conj, show, n_show) -> None:
    """모델 입력 인자만 key: value 로. 주석 없음."""
    H, act, txv = T["H"], T["active_bins"], T["tx_valid"]
    rssi, snr, t = T["rssi"], T["snr"], T["t"]
    nrx = T["nrx"]

    _kv("t", "%.3f s" % t[i])
    _kv("active_bins", "%d / %d" % (len(act), H.shape[-1]))
    _kv("rssi", "%s dBm" % list(map(int, rssi[i])))
    _kv("snr", "%s dB" % list(map(int, snr[i])))
    # Nc0 은 모든 패킷에 존재하므로 모델은 이것만 쓰면 된다.
    for rx in range(nrx):
        _kv("V[k,Nr%d]" % rx, " ".join(_cx(H[i, rx, 0, k]) for k in show))
    if conj is None:
        _kv("CSI_ratio[k]", "(Rx 1경로라 계산 불가)")
    else:
        _kv("CSI_ratio[k]", " ".join(_cx(conj[i, 0, 0, k]) for k in show))

    bad = []
    for k in p.chains:
        if k[0] != 0:
            continue
        z = p.chains[k].csi_np()
        if not (np.array_equal(H[i, k[1], 0].real, z.real)
                and np.array_equal(H[i, k[1], 0].imag, z.imag)):
            bad.append(k)
    vtx = [tx for tx in range(T["ntx"]) if txv[i, tx]]
    agc_ok = all(abs(20 * np.log10(float(np.sqrt((np.abs(np.concatenate(
        [Hg[i, rx, tx, act] for tx in vtx])) ** 2).mean())) + 1e-30)
        - rssi[i, rx]) < 0.01 for rx in range(nrx))
    if not bad and agc_ok:
        _kv("check", "OK")
    else:
        _kv("check", "FAIL " + " ".join(
            ([] if not bad else ["tlv!=tensor%s" % bad])
            + ([] if agc_ok else ["agc!=rssi"])))


def sample_dump(cap: Capture, interval: float = 1.0, n_show: int = 4,
                pace: bool = False, limit: Optional[int] = None,
                model: bool = False) -> int:
    """일정 간격으로 패킷을 골라 모든 인자를 key: value 로 찍는다.

    요약 통계(verify)는 통과하는데 실제 값이 이상한 경우를 잡기 위한 것이다.
    원시 TLV(체인별)와 텐서/특징 값(모델이 실제로 받는 것)을 나란히 보여줘서
    둘 사이에 어긋남이 없는지 눈으로 대조할 수 있게 한다.

    interval : 캡처 시간 기준 표본 간격 [s]. 0 이면 전 패킷.
    pace     : 실제로 interval 만큼 쉬면서 출력 (스크롤 속도 맞추기).
    """
    _need_numpy()
    packets, gs = group_chains(cap)
    if not packets:
        print("패킷이 없습니다.")
        return 1
    T = tensor_from_packets(packets)
    t, sel = T["t"], T["packets"]
    H, act, txv = T["H"], T["active_bins"], T["tx_valid"]
    rssi, snr, seq = T["rssi"], T["snr"], T["seq"]
    Hg = apply_agc(H, rssi, txv, act)
    nrx, ntx = T["nrx"], T["ntx"]
    conj = feature_conj(H) if nrx >= 2 else None
    ratio = feature_ratio(H) if nrx >= 2 else None

    idx: List[int] = []
    nxt = 0.0
    for i, tv in enumerate(t):
        if interval <= 0 or tv + 1e-9 >= nxt:
            idx.append(i)
            nxt = tv + interval
    if limit:
        idx = idx[:limit]

    dur = float(t[-1]) if len(t) > 1 else 0.0
    rate = (len(t) - 1) / dur if dur > 0 else 0.0
    print("=" * 78)
    print(" csitool sampletest%s%s"
          % (" " * 40, ("csi_parse v" + __version__).rjust(19)))
    print(" %-11s %s" % ("file", cap.source or "<stdin>"))
    print(" %-11s %d packets   %.1f s   %.1f Hz"
          % ("capture", len(t), dur, rate))
    print(" %-11s %d/%d subcarriers   %d Rx x %d Tx   %d MHz %s"
          % ("layout", len(act), H.shape[-1], nrx, ntx,
             T["bw_mhz"], FRAME_MODE.get(T["frame_mode"], "?")))
    print(" %-11s %s" % ("peer", T["peer"] or "(unfiltered)"))
    print(" %-11s %d samples @ %.2f s%s"
          % ("sampling", len(idx), interval,
             "  (paced)" if pace else ""))
    print("=" * 78)

    show = [int(a) for a in act[:n_show]]
    for n, i in enumerate(idx):
        p = sel[i]
        head = " sample %d/%d   t=+%.3f s   seq=%d " % (n + 1, len(idx),
                                                          t[i], seq[i])
        print("\n--" + head + "-" * max(0, 76 - len(head)))

        if model:
            _sample_model(i, p, T, Hg, conj, show, n_show)
            if pace and n + 1 < len(idx):
                time.sleep(interval)
            continue

        print("  [패킷]")
        _kv("t", "%.3f" % t[i], "s", "절대 %.3f s" % T["t_abs"][i])
        _kv("seq", int(seq[i]))
        _kv("tx_mac", p.tx_mac)
        _kv("bandwidth", p.bw, "", "DBW%d, %d 부반송파"
            % (BW_MHZ.get(p.bw, 0), BW_NSUB.get(p.bw, 0)))
        _kv("frame_mode", p.frame_mode, "", FRAME_MODE.get(p.frame_mode, "?"))
        _kv("nsub", p.nsub)
        _kv("active_bins", "%d 개" % len(act), "",
            "bin %d..%d" % (int(act[0]), int(act[-1])))
        _kv("tx_valid", list(map(bool, txv[i])), "",
            "%dSS" % int(txv[i].sum()))
        _kv("rssi", list(map(int, rssi[i])), "dBm", "rx0..rx%d" % (nrx - 1))
        _kv("snr", list(map(int, snr[i])), "dB")

        print("  [체인별 원시 TLV]")
        for (tx, rx) in sorted(p.chains):
            c = p.chains[(tx, rx)]
            I, Q = c.csi_int()
            print("    chain(tx%d,rx%d)  ver=%d pri_ch=%d extra=0x%08x "
                  "seq_chain=%d last=%d rssi=%d snr=%d"
                  % (tx, rx, c.version, c.pri_ch, c.extra, c.seq_chain,
                     c.seq_last, c.rssi, c.snr))
            print("      csi_i[%s] = %s" % (show, [I[k] for k in show]))
            print("      csi_q[%s] = %s" % (show, [Q[k] for k in show]))

        print("  [텐서 - 모델이 받는 값]")
        for rx in range(nrx):
            for tx in range(ntx):
                if not txv[i, tx]:
                    _kv("H[rx%d,tx%d]" % (rx, tx), "(tx 미사용)")
                    continue
                _kv("H[rx%d,tx%d]" % (rx, tx),
                    " ".join(_cx(H[i, rx, tx, k]) for k in show))
        # 통계는 그 패킷에서 실제로 쓰인 Tx 슬롯 전체에 대해 낸다.
        # tx0 만 보면 2SS 패킷에서 AGC 항등식(RMS == RSSI)이 어긋나 보인다.
        vtx = [tx for tx in range(ntx) if txv[i, tx]]
        for rx in range(nrx):
            v = np.concatenate([H[i, rx, tx, act] for tx in vtx])
            _kv("|H| rx%d" % rx, "RMS %.1f  min %.1f  max %.1f"
                % (float(np.sqrt((np.abs(v) ** 2).mean())),
                   float(np.abs(v).min()), float(np.abs(v).max())),
                "", "유효 tx %s" % vtx)
        for rx in range(nrx):
            v = np.concatenate([Hg[i, rx, tx, act] for tx in vtx])
            db = 20 * np.log10(float(np.sqrt((np.abs(v) ** 2).mean())) + 1e-30)
            _kv("H_agc rx%d" % rx, "RMS %.2f dB" % db, "",
                "RSSI %d dBm  %s" % (rssi[i, rx],
                                     "일치" if abs(db - rssi[i, rx]) < 0.01
                                     else "<-- 어긋남!"))

        bad = []
        for (tx, rx), c in p.chains.items():
            I, Q = c.csi_np().real, c.csi_np().imag
            if not (np.array_equal(H[i, rx, tx].real, I)
                    and np.array_equal(H[i, rx, tx].imag, Q)):
                bad.append("tx%drx%d" % (tx, rx))
        _kv("원시TLV <-> 텐서", "일치 (%d chain x %d 부반송파)"
            % (len(p.chains), p.nsub) if not bad else "불일치 %s" % bad)

        print("  [파생 특징]")
        if conj is None:
            _kv("conj / ratio", "(Rx 1경로라 계산 불가)")
        else:
            for tx in range(ntx):
                if not txv[i, tx]:
                    _kv("conj[tx%d]" % tx, "(tx 미사용)")
                    continue
                _kv("conj[tx%d]" % tx,
                    " ".join(_cx(conj[i, 0, tx, k]) for k in show))
                _kv("  |conj|/arg", " ".join(
                    "%.3g@%+.2f" % (abs(conj[i, 0, tx, k]),
                                    float(np.angle(conj[i, 0, tx, k])))
                    for k in show), "rad")
                _kv("ratio[tx%d]" % tx,
                    " ".join(_cx(ratio[i, 0, tx, k], 3) for k in show))
        if i > 0 and conj is not None:
            a = conj[i - 1, 0, 0, act]
            b = conj[i, 0, 0, act]
            m = np.isfinite(a) & np.isfinite(b) & (np.abs(a) > 0) & (np.abs(b) > 0)
            if m.any():
                d = (b * np.conj(a))[m]
                _kv("직전 표본과 결맞음", "%.3f" % float(np.abs((d / np.abs(d)).mean())),
                    "", "표본 간격이 넓으면 낮게 나오는 게 정상")

        if pace and n + 1 < len(idx):
            time.sleep(interval)
    print()
    return 0


# =============================================================================
# 7. CLI
# =============================================================================

def _cmd_info(a) -> int:
    cap = load_capture(a.file)
    packets, gs = group_chains(cap)
    print("파일        : %s (%d 바이트)" % (a.file, cap.file_bytes))
    print("백엔드      : %s" % cap.backend)
    print("레코드      : %d (resync %d B, trailing %d B)"
          % (len(cap.chains), cap.resync_bytes, cap.trailing_bytes))
    if not cap.chains:
        return 1
    c0 = cap.chains[0]
    print("패킷        : %d (그룹핑 키=%s, 불완전 %d)"
          % (len(packets), gs.get("key"), gs.get("orphan", 0)))
    print("대역폭      : DBW%d (%d 부반송파)" % (BW_MHZ.get(c0.bw, 0), c0.nsub))
    print("프레임 모드 : %s" % FRAME_MODE.get(c0.frame_mode, "?"))
    print("송신 AP     : %s" % c0.tx_mac)
    rx = sorted({c.rx_idx for c in cap.chains})
    tx = sorted({c.tx_idx for c in cap.chains})
    print("경로        : Tx %s x Rx %s" % (tx, rx))
    if packets:
        dur = (packets[-1].ts_ms - packets[0].ts_ms) / 1000.0
        print("구간        : %.1f s, 실효 %.1f Hz"
              % (dur, (len(packets) - 1) / dur if dur > 0 else 0))
    rs = [c.rssi for c in cap.chains]
    sn = [c.snr for c in cap.chains]
    print("RSSI / SNR  : %d..%d dBm / %d..%d dB"
          % (min(rs), max(rs), min(sn), max(sn)))
    return 0


def _cmd_verify(a) -> int:
    cap = load_capture(a.file)
    v = Verifier(cap)
    v.run()
    rc = print_report(v, cap)
    if a.census:
        print_census(v)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"file": a.file, "backend": cap.backend,
                       "summary": v.summary(),
                       "checks": [vars(c) for c in v.checks]},
                      f, ensure_ascii=False, indent=2)
        print("JSON 저장: %s" % a.json)
    return rc


def _cmd_dump(a) -> int:
    cap = load_capture(a.file)
    dump_records(cap, a.n)
    return 0


def _cmd_export(a) -> int:
    T = tensor_from_packets(group_chains(load_capture(a.file))[0],
                            peer=None if a.peer == "none" else a.peer)
    d = export_npz(T, a.out, a.feature)
    print("저장: %s" % a.out)
    for k in sorted(d):
        val = d[k]
        shape = getattr(val, "shape", None)
        print("  %-20s %s %s" % (k, shape if shape is not None else "",
                                 getattr(val, "dtype", type(val).__name__)))
    return 0


def _cmd_sampletest(a) -> int:
    return sample_dump(load_capture(a.file), a.interval, a.n_show,
                       a.pace, a.limit, a.model)


def _cmd_selftest(a) -> int:
    return selftest(verbose=not a.quiet)


def _cmd_capture(a) -> int:
    return device_capture(a.out, a.sec, a.iface, a.ap, a.frame_type,
                          a.iwpriv, a.band, a.ping_interval)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="csitool.py",
        description="Wi-Fi CSI 진단 도구 (csi_parse v%s)" % __version__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("사용법")[-1])
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_file(sp):
        sp.add_argument("file")

    s = sub.add_parser("info", help="캡처 요약")
    add_file(s); s.set_defaults(func=_cmd_info)

    s = sub.add_parser("verify", help="파싱 정합성 전수 검사")
    add_file(s)
    s.add_argument("--json", default=None, help="검사 결과를 JSON 으로 저장")
    s.add_argument("--census", action="store_true", help="TLV 태그 census 출력")
    s.set_defaults(func=_cmd_verify)

    s = sub.add_parser("dump", help="TLV 주석 덤프")
    add_file(s)
    s.add_argument("-n", type=int, default=3, help="출력할 레코드 수")
    s.set_defaults(func=_cmd_dump)

    s = sub.add_parser("export", help="정규 텐서를 .npz 로 저장")
    add_file(s)
    s.add_argument("-o", "--out", required=True)
    s.add_argument("--feature", default="all",
                   choices=["all", "ratio", "conj", "agc", "none"])
    s.add_argument("--peer", default="auto",
                   help="이 송신자 MAC 만 사용 ('auto'=최다 출현, 'none'=필터 안함)")
    s.set_defaults(func=_cmd_export)

    s = sub.add_parser("sampletest",
                       help="일정 간격으로 파싱된 전 인자를 key: value 로 출력")
    add_file(s)
    s.add_argument("-i", "--interval", type=float, default=1.0,
                   help="표본 간격[s]. 0 이면 전 패킷 (기본 1.0)")
    s.add_argument("-n", "--n-show", type=int, default=4,
                   help="표본마다 보여줄 부반송파 개수 (기본 4)")
    s.add_argument("--pace", action="store_true",
                   help="interval 만큼 실제로 쉬면서 출력")
    s.add_argument("--limit", type=int, default=None, help="최대 표본 수")
    s.add_argument("--model", action="store_true",
                   help="모델 입력 인자만 제안서 용어로 표시 (원시 TLV 생략)")
    s.set_defaults(func=_cmd_sampletest)

    s = sub.add_parser("selftest", help="회귀 검사 (합성 데이터)")
    s.add_argument("-q", "--quiet", action="store_true")
    s.set_defaults(func=_cmd_selftest)

    s = sub.add_parser("capture", help="온디바이스 캡처 (MediaTek)")
    s.add_argument("-o", "--out", required=True)
    s.add_argument("--sec", type=int, default=20)
    s.add_argument("--iface", default="wlan0")
    s.add_argument("--ap", default=None, help="ping 대상 AP IP (트래픽 생성)")
    s.add_argument("--frame-type", type=int, default=34,
                   help="34=QoS Data(권장), 32=Beacon")
    s.add_argument("--band", type=int, default=0)
    s.add_argument("--iwpriv", default="iwpriv", help="iwpriv 실행 경로")
    s.add_argument("--ping-interval", type=float, default=0.02,
                   help="트래픽 생성 ping 간격[s]. 샘플레이트를 결정한다 (기본 0.02=50Hz)")
    s.set_defaults(func=_cmd_capture)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print("오류: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())