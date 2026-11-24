"""Configuration registry of the end-to-end study.

Round 1 (pre-registered before any result): five architectures x calibration none (A) / per-bin fixed5 noise
normalisation (B), default training settings. S = each window by its own per-bin noise level (regime A).
Round 2+ (chosen from INNER scores only): single-factor variations of a base config, named <arch>.<cal>.<tag>[.<tag>].
"""
from __future__ import annotations

from dataclasses import replace

from e2e import Cfg

BASE = {
    "cnn1d": Cfg("cnn1d", arch="cnn1d", width=64, depth=5),
    "scnet": Cfg("scnet", arch="scnet", width=64, depth=5),
    "spec": Cfg("spec", arch="spec", width=64, depth=5, mix_k=8),
    "conv2d": Cfg("conv2d", arch="conv2d", width=64, depth=5),
    "patchtf": Cfg("patchtf", arch="patchtf", width=64, layers=2, lr=5e-4),
    "sbo": Cfg("sbo", arch="sbo", width=64, depth=5),
}

REG: dict[str, Cfg] = {}
for k, c in BASE.items():
    REG[f"{k}.A"] = replace(c, name=f"{k}.A", cal="none")
    REG[f"{k}.B"] = replace(c, name=f"{k}.B", cal="persc")
    REG[f"{k}.S"] = replace(c, name=f"{k}.S", cal="self")
    REG[f"{k}.C"] = replace(c, name=f"{k}.C", cal="scalar")

VARS = {
    "w32": dict(width=32), "w96": dict(width=96), "d05": dict(drop=0.5), "wd2": dict(wd=0.2),
    "e20": dict(epochs=20), "e80": dict(epochs=80), "c20": dict(crop=800), "c60": dict(crop=2400),
    "st8": dict(stride=8), "st2": dict(stride=2), "sc3": dict(aug_scale=0.3), "ls1": dict(label_smooth=0.1),
    "ema": dict(ema=0.99), "q": dict(binpool="q"), "ac": dict(acpool=30), "adv": dict(adv=0.3),
    "ref": dict(refmode="diff"), "mae": dict(ssl="mae"), "sim": dict(ssl="simclr"), "gn": dict(norm="gn"),
    "noaug": dict(aug_rev=False, aug_conj=False, aug_rot=False), "lr3": dict(lr=3e-4), "d4": dict(depth=4),
    "norev": dict(aug_rev=False), "noconj": dict(aug_conj=False), "fcrop": dict(rcrop=False), "foff": dict(roff=False),
    "d6": dict(depth=6), "k5": dict(kernel=5), "k11": dict(kernel=11),
    "rg": dict(inp="rg"), "g": dict(inp="g"),
    "xt": dict(xtrain=True),
    "auxp2": dict(aux=2.0, aux_ch="phys13"), "auxp5": dict(aux=5.0, aux_ch="phys13"),
    "bnj": dict(aux=2.0, aux_ch="phys13", bneck="joint"), "bnd": dict(aux=2.0, aux_ch="phys13", bneck="detach"),
    "bnj17": dict(aux=2.0, aux_ch="seq17", bneck="joint"), "bnja": dict(aux=2.0, aux_ch="all", bneck="joint"),
    "lf": dict(specbr="lf"), "hf": dict(specbr="hf"), "lfhf": dict(specbr="lf+hf"),
    "lfhftf": dict(specbr="lf+hf+tf"), "lfhftl": dict(specbr="lf+hf+tl"), "all4": dict(specbr="lf+hf+tf+tl"),
    "st16": dict(stride=16), "st32": dict(stride=32), "st64": dict(stride=64),
    "tftl": dict(specbr="tf+tl"), "t4": dict(specbr="tf+tl+ts+tm"), "all6": dict(specbr="lf+hf+tf+tl+ts+tm"),
    "s7": dict(sbstat=7), "sw2": dict(sbw=2), "att": dict(sbatt=True),
    "coh": dict(specbr="clf+cmf+chf"), "all4c": dict(specbr="lf+hf+tf+tl+clf+cmf+chf"),
    "lfhfc": dict(specbr="lf+hf+clf+cmf+chf"),
    "aux": dict(aux=0.5), "aux2": dict(aux=2.0), "auxp": dict(aux=0.5, aux_ch="phys13"), "auxs": dict(aux=0.5, aux_ch="seq17"),
}


def get(name: str) -> Cfg:
    if name in REG:
        return REG[name]
    parts = name.split(".")
    cfg = REG[".".join(parts[:2])]
    for tag in parts[2:]:
        cfg = replace(cfg, **VARS[tag])
    return replace(cfg, name=name)
