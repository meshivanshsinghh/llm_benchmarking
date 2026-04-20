from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("charts")

FILES = {
    "a10g":        RESULTS_DIR / "all_results_a10G.json",
    "h100_vllm":   RESULTS_DIR / "vllm_h100.json",
    "h100_sglang": RESULTS_DIR / "sglang_h100.json",
}

COLORS = {
    "vllm":   "#C1292E",    
    "sglang": "#235789",    
    "ollama": "#7D7D7D",
}

TINT = {
    "vllm":   "#E89194",
    "sglang": "#7FA3C7",
    "ollama": "#C4C4C0",
}

# Canvas
BG    = "#FAFAF7" 
FG    = "#1F1F1F" 
GRID  = "#E5E5E0"
MUTED = "#8A8A85"

LABEL = {"vllm": "vLLM", "sglang": "SGLang", "ollama": "Ollama"}

SOURCE_LINE = (
    "Llama 3.1 8B AWQ-INT4 · 300 ShareGPT prompts · max 128 output tokens · "
    "10 warmup requests · single run per config."
)

def pick_font() -> str:
    preferred = [
        "Inter", "IBM Plex Sans", "Source Sans 3", "Source Sans Pro",
        "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((f for f in preferred if f in available), "DejaVu Sans")
    if chosen == "DejaVu Sans":
        print("[warn] Inter/Helvetica not found — using DejaVu Sans. "
              "For the editorial look install Inter from Google Fonts and "
              "clear ~/.cache/matplotlib.")
    else:
        print(f"[info] using font: {chosen}")
    return chosen


def setup_style():
    font = pick_font()
    plt.rcParams.update({
        "font.family": font,
        "font.size": 11,
        "text.color": FG,

        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": FG,
        "axes.linewidth": 0.7,
        "axes.labelcolor": FG,
        "axes.labelsize": 10.5,
        "axes.labelpad": 8,
        "axes.titlesize": 15,
        "axes.titleweight": "semibold",
        "axes.titlepad": 16,
        "axes.titlelocation": "left",

        # NYT/editorial: strip all spines except the bottom axis.
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.spines.left":  False,

        "xtick.color": FG,
        "ytick.color": FG,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.major.size": 4,
        "xtick.major.width": 0.6,
        "ytick.major.size": 0,

        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.axisbelow": True,

        "legend.frameon": False,
        "legend.fontsize": 10,

        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": BG,
        "savefig.edgecolor": "none",
    })

def _normalize_entry(entry: dict) -> dict:
    """Flatten one per-concurrency record; tolerate minor key renames."""
    return {
        "concurrency": entry["concurrency"],
        "throughput":  entry.get("throughput", entry.get("system_tps")),
        "ttft_p50":    entry.get("ttft_p50"),
        "ttft_p95":    entry.get("ttft_p95"),
        "ttft_p99":    entry.get("ttft_p99"),
        "tpot_p50":    entry.get("tpot_p50"),
        "e2e_p50":     entry.get("e2e_p50"),
        "e2e_p99":     entry.get("e2e_p99"),
        "rps":         entry.get("requests_per_sec", entry.get("rps")),
        "success":     entry.get("success_rate", entry.get("success", 100.0)),
    }


def _transpose(records: list[dict]) -> dict:
    records = sorted(records, key=lambda r: r["concurrency"])
    keys = records[0].keys()
    return {k: [r[k] for r in records] for k in keys}


def _parse_file(path: Path) -> dict[str, dict]:
    """Return {framework: columnar dict}. Handles two JSON layouts."""
    raw = json.loads(path.read_text())
    by_fw: dict[str, list[dict]] = {}

    # Layout A: {"results": [{"framework": ..., "concurrency": ..., ...}, ...]}
    if isinstance(raw, dict) and "results" in raw:
        for entry in raw["results"]:
            fw = entry["framework"].lower()
            by_fw.setdefault(fw, []).append(_normalize_entry(entry))

    # Layout B: {"vllm": {...}, "sglang": {...}, "ollama": {...}}
    #          where each value is either columnar or a list of records.
    elif isinstance(raw, dict) and any(
        k in raw for k in ("vllm", "sglang", "ollama")
    ):
        for fw_raw, block in raw.items():
            fw = fw_raw.lower()
            if fw not in ("vllm", "sglang", "ollama"):
                continue
            if isinstance(block, dict) and "concurrency" in block:
                n = len(block["concurrency"])
                by_fw[fw] = [
                    _normalize_entry({k: v[i] for k, v in block.items()
                                      if isinstance(v, list) and len(v) == n}
                                     | {"concurrency": block["concurrency"][i]})
                    for i in range(n)
                ]
            elif isinstance(block, list):
                by_fw[fw] = [_normalize_entry(r) for r in block]
    else:
        raise ValueError(f"Unrecognized JSON shape: {path}")

    return {fw: _transpose(recs) for fw, recs in by_fw.items()}


def load_data() -> dict:
    out: dict = {"A10G": {}, "H100": {}}
    
    for key, dest in [("a10g", "A10G"),
                      ("h100_vllm", "H100"),
                      ("h100_sglang", "H100")]:
        path = FILES[key]
        try:
            out[dest].update(_parse_file(path))
        except Exception as e:
            print(f'error: {e}')    
    return out

def save(fig, name: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / f"{name}.png"
    svg = OUTPUT_DIR / f"{name}.svg"
    fig.savefig(png)
    fig.savefig(svg)
    plt.close(fig)
    print(f"  saved {png}")


def add_subtitle(fig, subtitle: str):
    """Small gray subtitle line directly below the axes title."""
    fig.text(0.125, 0.895, subtitle, fontsize=11, color=MUTED,
             style="italic", ha="left", va="top")


def add_source(fig, text: str = SOURCE_LINE):
    fig.subplots_adjust(bottom=0.17)
    fig.text(0.5, 0.04, text, fontsize=8.5, color=MUTED, ha="center", va="bottom")


def annotate_end(ax, x, y, label, color, dx=0.01, dy=0, weight="semibold"):
    """Direct end-of-line label, right of the last point."""
    ax.annotate(
        label, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
        color=color, fontsize=10.5, fontweight=weight,
        va="center", ha="left",
    )

def thousands(x, _):
    if x >= 1000:
        return f"{x/1000:.1f}k".rstrip("0").rstrip(".") + "k" if False else f"{int(x):,}"
    return f"{x:g}"

def chart_01_throughput_a10g(data):
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for fw in ("sglang", "vllm", "ollama"):
        d = data["A10G"][fw]
        ax.plot(d["concurrency"], d["throughput"],
                color=COLORS[fw], linewidth=2.2, marker="o",
                markersize=5.5, markerfacecolor=COLORS[fw],
                markeredgecolor=BG, markeredgewidth=1.5,
                zorder=3 if fw == "sglang" else 2)
        annotate_end(ax, d["concurrency"][-1], d["throughput"][-1],
                     f"  {LABEL[fw]}", COLORS[fw])

    ax.set_yscale("log")
    ax.set_xlabel("Concurrent requests")
    ax.set_ylabel("Output tokens / second")
    ax.set_title("SGLang leads at every load level on the A10G")
    ax.set_xticks([1, 8, 32, 64, 128])
    ax.set_xlim(0.5, 180)
    ax.yaxis.set_major_formatter(FuncFormatter(thousands))

    # Callout moved to the empty space between the Ollama cluster and vLLM line
    ax.annotate(
        "Ollama collapses above\n8 concurrent users",
        xy=(32, 0.20), xytext=(48, 2.2),
        fontsize=9.5, color=MUTED, style="italic", ha="center",
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6,
                        connectionstyle="arc3,rad=0.25"),
    )

    add_subtitle(fig, "Throughput vs concurrent requests · A10G (Ampere, 24 GB) · log scale")
    add_source(fig)
    save(fig, "01_throughput_a10g")


def chart_02_throughput_h100(data):
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for fw in ("sglang", "vllm"):
        d = data["H100"][fw]
        ax.plot(d["concurrency"], d["throughput"],
                color=COLORS[fw], linewidth=2.2, marker="o",
                markersize=5.5, markerfacecolor=COLORS[fw],
                markeredgecolor=BG, markeredgewidth=1.5)
        annotate_end(ax, d["concurrency"][-1], d["throughput"][-1],
                     f"  {LABEL[fw]}", COLORS[fw])

    ax.set_xlabel("Concurrent requests")
    ax.set_ylabel("Output tokens / second")
    ax.set_title("On Hopper, SGLang's lead widens to 3.4×")
    ax.set_xticks([1, 8, 32, 64, 128])
    ax.set_xlim(0.5, 165)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Note about Ollama
    ax.text(0.98, 0.02,
            "Ollama excluded: GPU detection failed on the rental instance.\n"
            "Its limitation is architectural (fixed-slot parallelism)",
            transform=ax.transAxes, fontsize=9, color=MUTED, style="italic",
            ha="right", va="bottom")

    add_subtitle(fig, "Throughput vs concurrent requests · H100 (Hopper, 80 GB)")
    add_source(fig)
    save(fig, "02_throughput_h100")

def _pareto(ax, data_gpu, frameworks):
    """Throughput vs P99 latency — each point = one concurrency level."""
    def _fmt(v, _):
        if v >= 1:
            return f"{int(v):,}"
        elif v >= 0.1:
            return f"{v:.1f}"
        else:
            return f"{v:.2f}"

    end_offset = {
        "vllm":   (10, 0),
        "sglang": (10, 0),
        "ollama": (10, 18),
    }

    for fw in frameworks:
        d = data_gpu[fw]
        ax.plot(d["throughput"], d["e2e_p99"],
                color=COLORS[fw], linewidth=1.8, alpha=0.7, zorder=2)
        ax.scatter(d["throughput"], d["e2e_p99"],
                   color=COLORS[fw], s=42, zorder=3,
                   edgecolor=BG, linewidth=1.2)

        x0, y0 = d["throughput"][0], d["e2e_p99"][0]
        ax.annotate("c=1", (x0, y0), xytext=(5, -14),
                    textcoords="offset points",
                    fontsize=8.5, color=MUTED, ha="left")

        dx, dy = end_offset.get(fw, (10, 0))
        annotate_end(ax, d["throughput"][-1], d["e2e_p99"][-1],
                     f"  {LABEL[fw]} (c={d['concurrency'][-1]})",
                     COLORS[fw], dx=dx, dy=dy)

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Throughput (tokens / second, higher is better)")
    ax.set_ylabel("P99 end-to-end latency (ms, lower is better)")
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt))
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt))
    
def chart_03_pareto_a10g(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    _pareto(ax, data["A10G"], ("sglang", "vllm", "ollama"))
    ax.set_title("Throughput - latency trade-off on A10G")
    ax.set_xticks([0.1, 1, 10, 100, 1000])
    ax.set_yticks([1000, 3000, 10000, 30000, 100000])
    add_subtitle(fig, "P99 end-to-end latency vs throughput · each point is a concurrency level · both axes log")
    add_source(fig)
    save(fig, "03_pareto_a10g")


def chart_04_pareto_h100(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    _pareto(ax, data["H100"], ("sglang", "vllm"))
    ax.set_title("Throughput - latency trade-off on H100")
    ax.set_xticks([100, 300, 1000, 3000, 10000])
    ax.set_yticks([400, 700, 1000, 2000, 3000, 5000])
    add_subtitle(fig, "P99 end-to-end latency vs throughput · each point is a concurrency level · both axes log")
    add_source(fig)
    save(fig, "04_pareto_h100")


def chart_05_cross_gpu_scaling(data):
    fig, ax = plt.subplots(figsize=(10, 5.6))

    frameworks = ["vllm", "sglang"]
    a10g_peak = [max(data["A10G"][fw]["throughput"]) for fw in frameworks]
    h100_peak = [max(data["H100"][fw]["throughput"]) for fw in frameworks]
    speedups = [h / a for h, a in zip(h100_peak, a10g_peak)]

    y = np.arange(len(frameworks))
    h = 0.34

    # A10G bars (tinted) and H100 bars (solid)
    for i, fw in enumerate(frameworks):
        ax.barh(y[i] + h/2, a10g_peak[i], height=h,
                color=TINT[fw], edgecolor=COLORS[fw], linewidth=0.8)
        ax.barh(y[i] - h/2, h100_peak[i], height=h,
                color=COLORS[fw])

        # End-of-bar value labels
        ax.text(a10g_peak[i] + 80, y[i] + h/2,
                f"{a10g_peak[i]:,.0f} tok/s  · A10G",
                va="center", fontsize=10, color=FG)
        ax.text(h100_peak[i] + 80, y[i] - h/2,
                f"{h100_peak[i]:,.0f} tok/s  · H100",
                va="center", fontsize=10, color=FG, fontweight="semibold")

        # Speedup callout, between bars
        ax.text(max(a10g_peak[i], h100_peak[i]) * 1.22, y[i],
                f"{speedups[i]:.1f}× scaling",
                va="center", fontsize=11, color=COLORS[fw],
                fontweight="semibold")

    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[fw] for fw in frameworks], fontsize=12,
                       fontweight="semibold")
    ax.set_xlim(0, max(h100_peak) * 1.55)
    ax.set_xlabel("Peak throughput at concurrency = 128  (tokens / second)")
    ax.set_title("SGLang benefits more from Hopper — 5.4× vs 2.5× for vLLM")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.invert_yaxis()

    add_subtitle(fig, "Peak throughput on A10G vs H100 · same model, same workload, only the GPU changes")
    add_source(fig)
    save(fig, "05_cross_gpu_scaling")


def chart_06_single_request_latency(data):
    """The 'what a user feels' chart — c=1 E2E P50 latency, horizontal bars."""
    fig, ax = plt.subplots(figsize=(10, 5.2))

    rows = [
        ("H100", "sglang", data["H100"]["sglang"]["e2e_p50"][0]),
        ("A10G", "sglang", data["A10G"]["sglang"]["e2e_p50"][0]),
        ("H100", "vllm",   data["H100"]["vllm"]["e2e_p50"][0]),
        ("A10G", "vllm",   data["A10G"]["vllm"]["e2e_p50"][0]),
        ("A10G", "ollama", data["A10G"]["ollama"]["e2e_p50"][0]),
    ]
    y = np.arange(len(rows))
    values = [r[2] for r in rows]
    colors = [COLORS[r[1]] if r[0] == "H100" else TINT[r[1]] for r in rows]
    edges  = [COLORS[r[1]] for r in rows]
    labels = [f"{LABEL[r[1]]}  ·  {r[0]}" for r in rows]

    bars = ax.barh(y, values, color=colors, edgecolor=edges, linewidth=0.8,
                   height=0.62)
    for bar, v in zip(bars, values):
        ax.text(v + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:,.0f} ms", va="center", fontsize=10.5, color=FG)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("End-to-end P50 latency for a single request (ms)")
    ax.set_title("What a user actually feels — single-request latency")
    ax.set_xlim(0, max(values) * 1.18)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Reference band only — label is now in the subtitle, no inline text
    ax.axvspan(0, 1000, color="#235789", alpha=0.06, zorder=0)

    add_subtitle(fig,
        "E2E P50 at concurrency = 1 · solid = H100, tinted = A10G · "
        "blue band = interactive zone (<1 second)")
    add_source(fig)
    save(fig, "06_single_request_latency")

def chart_07_ttft_tail_c128(data):
    fig, ax = plt.subplots(figsize=(10, 4.8))

    # Ordered best P99 → worst P99, top to bottom
    combos = [
        ("SGLang · H100", "sglang", "H100"),
        ("vLLM · H100",   "vllm",   "H100"),
        ("vLLM · A10G",   "vllm",   "A10G"),
        ("SGLang · A10G", "sglang", "A10G"),
    ]

    values = [data[gpu][fw]["ttft_p99"][-1] for _, fw, gpu in combos]
    colors = [COLORS[fw] if gpu == "H100" else TINT[fw]
              for _, fw, gpu in combos]
    edges  = [COLORS[fw] for _, fw, gpu in combos]
    labels = [c[0] for c in combos]

    y = np.arange(len(combos))
    bars = ax.barh(y, values, color=colors, edgecolor=edges,
                   linewidth=0.8, height=0.58)

    for bar, v in zip(bars, values):
        ax.text(v + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{v:,.0f} ms", va="center", fontsize=11, color=FG)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11.5)
    ax.invert_yaxis()
    ax.set_xlabel("TTFT P99 at concurrency = 128  (ms)")
    ax.set_title("Tail latency at 128 concurrent users")
    ax.set_xlim(0, max(values) * 1.20)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

    add_subtitle(fig,
        "99th-percentile time to first token · c=128 · lower is better · "
        "solid = H100, tinted = A10G")
    add_source(fig)
    save(fig, "07_ttft_tail_c128")

def chart_08_ollama_collapse(data):
    """Two-panel: success rate (top) + E2E P50 latency (bottom) on A10G."""
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 6.8),
        gridspec_kw={"height_ratios": [1, 1.2], "hspace": 0.35},
        sharex=True,
    )

    d = data["A10G"]["ollama"]
    conc = d["concurrency"]

    # Top panel — success rate
    bar_colors = ["#3F8A4E" if s >= 95 else "#C1292E" for s in d["success"]]
    ax_top.bar(range(len(conc)), d["success"], width=0.55,
               color=bar_colors, edgecolor=BG, linewidth=1.2)
    for i, s in enumerate(d["success"]):
        ax_top.text(i, s + 3, f"{s:.1f}%", ha="center", fontsize=10, color=FG)
    ax_top.set_ylim(0, 115)
    ax_top.set_ylabel("Request success rate")
    ax_top.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax_top.set_title("Ollama collapses architecturally — not a tuning problem")

    # Bottom panel — E2E P50 latency (log)
    ax_bot.plot(range(len(conc)), d["e2e_p50"],
                color=COLORS["ollama"], linewidth=2.2,
                marker="o", markersize=6, markerfacecolor=COLORS["ollama"],
                markeredgecolor=BG, markeredgewidth=1.2)
    for i, v in enumerate(d["e2e_p50"]):
        label = f"{v/1000:.1f}s" if v < 60000 else f"{v/1000:.0f}s"
        ax_bot.text(i, v * 1.25, label, ha="center", fontsize=10,
                    color=FG)
    ax_bot.set_yscale("log")
    ax_bot.set_ylabel("End-to-end P50 latency")
    ax_bot.set_xlabel("Concurrent requests")
    ax_bot.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v/1000:.0f}s" if v >= 1000 else f"{int(v)}ms"))

    ax_bot.set_xticks(range(len(conc)))
    ax_bot.set_xticklabels([str(c) for c in conc])

    add_subtitle(fig,
        "Ollama on A10G · Llama 3.1 8B · OLLAMA_NUM_PARALLEL not set "
        "(default fixed-slot scheduling from llama.cpp)")
    add_source(fig)
    save(fig, "08_ollama_collapse")

def write_summary(data):
    summary = {
        "peak_throughput_tps": {
            gpu: {fw: max(data[gpu][fw]["throughput"])
                  for fw in data[gpu]}
            for gpu in data
        },
        "single_request_p50_ms": {
            gpu: {fw: data[gpu][fw]["e2e_p50"][0] for fw in data[gpu]}
            for gpu in data
        },
        "cross_gpu_speedup_c128": {
            fw: max(data["H100"][fw]["throughput"])
                / max(data["A10G"][fw]["throughput"])
            for fw in ("vllm", "sglang")
        },
        "sglang_vs_vllm_throughput_c128": {
            gpu: (data[gpu]["sglang"]["throughput"][-1]
                  / data[gpu]["vllm"]["throughput"][-1])
            for gpu in ("A10G", "H100")
        },
        "ollama_success_rate_a10g": dict(zip(
            data["A10G"]["ollama"]["concurrency"],
            data["A10G"]["ollama"]["success"],
        )),
    }
    path = OUTPUT_DIR / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"  wrote {path}")


def main():
    setup_style()
    data = load_data()
    
    chart_01_throughput_a10g(data)
    chart_02_throughput_h100(data)
    chart_03_pareto_a10g(data)
    chart_04_pareto_h100(data)
    chart_05_cross_gpu_scaling(data)
    chart_06_single_request_latency(data)
    chart_07_ttft_tail_c128(data)
    chart_08_ollama_collapse(data)
    
    write_summary(data)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)