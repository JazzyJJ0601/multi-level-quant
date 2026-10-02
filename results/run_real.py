#!/usr/bin/env python3
"""Real results: per-layer mixed 3/4-bit quantisation of Qwen3-8B at a fixed bit budget.

Every decoder linear gets 3 or 4 bits (group-wise RTN, group 128). At a budget where a
fraction F of the weights may use 4 bits, the method gives 4 bits to the layers where it
cuts the most activation-weighted quantisation error per weight (greedy, calibrated on
WikiText-2 train). The fair baseline is the same budget with layers picked at random.

Configs: fp16, uniform 4-bit, uniform 3-bit, and for F in {25%, 50%}: sensitivity-chosen
vs random-chosen (3 seeds). Perplexity on WikiText-2 test, 40 x 512 tokens.
Results go to results/real.json as each config finishes.
"""
import gc
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from qcommon import (act_stats, decoder_linears, load_model, packed_gb, perplexity, rtn,  # noqa: E402
                     save, weighted_err, windows)

OUT = HERE / "real.json"
N_TEST, N_CALIB = 40, 8


def allocate(linears, order, frac):
    """Give 4 bits to layers in `order` until `frac` of all linear weights are 4-bit."""
    total = sum(m.weight.numel() for _, m in linears)
    size = {n: m.weight.numel() for n, m in linears}
    bits, used = {n: 3 for n, _ in linears}, 0
    for n in order:
        if used + size[n] > frac * total:
            continue
        bits[n], used = 4, used + size[n]
    return bits, used / total


def main():
    model, tok = load_model()
    test, calib = windows(tok, "test", N_TEST), windows(tok, "train", N_CALIB)
    linears = decoder_linears(model)
    originals = {n: m.weight.data.to("cpu", copy=True) for n, m in linears}
    acts = act_stats(model, linears, calib)

    # sensitivity: error saved per weight by going 3 -> 4 bits
    gain = {}
    for n, m in linears:
        w = m.weight.data
        gain[n] = (weighted_err(w, rtn(w, 3), acts[n]) - weighted_err(w, rtn(w, 4), acts[n])) / w.numel()
    by_gain = sorted(gain, key=gain.get, reverse=True)

    configs = {"fp16": None, "uniform_4bit": {n: 4 for n, _ in linears},
               "uniform_3bit": {n: 3 for n, _ in linears}}
    for frac in (0.25, 0.5):
        configs[f"sensitivity_{int(frac * 100)}pct4bit"] = allocate(linears, by_gain, frac)[0]
        for seed in range(3):
            order = [n for n, _ in linears]
            random.Random(seed).shuffle(order)
            configs[f"random_{int(frac * 100)}pct4bit_seed{seed}"] = allocate(linears, order, frac)[0]

    results = {"setup": {"model": "Qwen3-8B", "group": 128, "quantiser": "asymmetric RTN",
                         "eval": f"WikiText-2 test, {N_TEST}x512 tokens",
                         "calib": f"WikiText-2 train, {N_CALIB}x512 tokens", "linears": len(linears)}}
    for name, bits in configs.items():
        t0 = time.time()
        for n, m in linears:
            m.weight.data = originals[n].cuda()
            if bits is not None:
                m.weight.data = rtn(m.weight.data, bits[n]).to(torch.bfloat16)
        row = {"ppl": round(perplexity(model, test), 4)}
        if bits is not None:
            sizes = {n: m.weight.numel() for n, m in linears}
            row["avg_bits"] = round(sum(bits[n] * sizes[n] for n in bits) / sum(sizes.values()), 3)
            row["packed_gb"] = packed_gb(model, bits)
            row["layers_4bit"] = sum(b == 4 for b in bits.values())
        row["seconds"] = round(time.time() - t0, 1)
        results[name] = row
        save(OUT, results)
        print(name, row, flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    top = by_gain[:20]
    results["most_sensitive_layers"] = top
    save(OUT, results)


if __name__ == "__main__":
    main()
