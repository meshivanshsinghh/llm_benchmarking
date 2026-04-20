import asyncio
import aiohttp
import json
import time
import random
import argparse
import os
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path
from datetime import datetime


def log(msg: str):
    """Print with flush — fixes invisible output when piped through nohup or tee."""
    print(msg, flush=True)


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class RequestResult:
    """Metrics for a single request."""
    request_id: int
    prompt_len: int
    output_len: int
    ttft_ms: float           # Time to First Token
    e2e_latency_ms: float    # End-to-end latency
    tpot_ms: float           # Time Per Output Token
    itl_ms: list             # Inter-Token Latencies
    output_tps: float        # Per-request tokens/sec
    success: bool
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Aggregate metrics for one (framework x concurrency) run."""
    framework: str
    model: str
    gpu: str
    concurrency: int
    num_requests: int
    duration_sec: float

    total_output_tokens: int = 0
    total_input_tokens: int = 0
    system_tps: float = 0.0
    requests_per_sec: float = 0.0

    ttft_p50: float = 0.0
    ttft_p95: float = 0.0
    ttft_p99: float = 0.0
    ttft_mean: float = 0.0

    tpot_p50: float = 0.0
    tpot_p95: float = 0.0
    tpot_p99: float = 0.0
    tpot_mean: float = 0.0

    e2e_p50: float = 0.0
    e2e_p95: float = 0.0
    e2e_p99: float = 0.0
    e2e_mean: float = 0.0

    itl_p50: float = 0.0
    itl_p95: float = 0.0
    itl_p99: float = 0.0
    itl_mean: float = 0.0

    success_rate: float = 0.0
    error_count: int = 0
    per_request: list = field(default_factory=list)


# ──────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────

def load_sharegpt_dataset(path: str, num_prompts: int = 1000) -> list:
    """
    Load ShareGPT — the standard dataset for LLM inference benchmarking.
    Real user conversations give realistic prompt/response length distributions.
    Falls back to synthetic prompts if the file isn't found.
    """
    prompts = []

    if os.path.exists(path):
        log(f"[DATA] Loading ShareGPT from {path}")
        with open(path, "r") as f:
            data = json.load(f)

        for conv in data:
            conversations = conv.get("conversations", [])
            if len(conversations) >= 2:
                human_msg = conversations[0].get("value", "")
                assistant_msg = conversations[1].get("value", "")

                if 10 < len(human_msg) < 4000 and 10 < len(assistant_msg) < 4000:
                    prompts.append({
                        "prompt": human_msg,
                        "expected_output_len": min(len(assistant_msg.split()) * 2, 512)
                    })

            if len(prompts) >= num_prompts:
                break

        log(f"[DATA] Loaded {len(prompts)} prompts from ShareGPT")

    # Fallback: synthetic prompts
    if len(prompts) < num_prompts:
        remaining = num_prompts - len(prompts)
        log(f"[DATA] Generating {remaining} synthetic prompts")

        templates = [
            "Explain the concept of {topic} in detail.",
            "Write a comparison between {topic} and its alternatives.",
            "Describe the step-by-step process of {topic}.",
            "What are the main challenges in {topic}?",
            "Summarize the current state of {topic} research.",
            "How does {topic} impact everyday life?",
            "Create a beginner's guide to {topic}.",
            "Analyze the future trends in {topic}.",
        ]

        topics = [
            "machine learning", "cloud computing", "data engineering",
            "distributed systems", "database optimization", "API design",
            "natural language processing", "computer vision", "edge AI",
            "recommendation systems", "time series analysis", "robotics",
        ]

        for i in range(remaining):
            prompt = random.choice(templates).format(topic=random.choice(topics))
            prompts.append({"prompt": prompt, "expected_output_len": random.randint(32, 256)})

    random.shuffle(prompts)
    return prompts[:num_prompts]


# ──────────────────────────────────────────────
# Core benchmark engine
# ──────────────────────────────────────────────

async def send_request(
    session: aiohttp.ClientSession,
    base_url: str,
    prompt: dict,
    request_id: int,
    model: str,
    max_tokens: int = 128,
    temperature: float = 0.0,
) -> RequestResult:
    """
    Send one streaming request and measure per-token timing.
    TTFT is measured client-side from the first SSE chunk containing content.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt["prompt"]}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    url = f"{base_url}/v1/chat/completions"
    token_timestamps = []
    output_text = ""
    first_token_time = None
    start_time = time.perf_counter()

    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                error_body = await resp.text()
                return RequestResult(
                    request_id=request_id,
                    prompt_len=len(prompt["prompt"].split()), output_len=0,
                    ttft_ms=0, e2e_latency_ms=0, tpot_ms=0,
                    itl_ms=[], output_tps=0,
                    success=False, error=f"HTTP {resp.status}: {error_body[:200]}"
                )

            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                    content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if content:
                        now = time.perf_counter()
                        token_timestamps.append(now)
                        output_text += content
                        if first_token_time is None:
                            first_token_time = now
                except json.JSONDecodeError:
                    continue

        end_time = time.perf_counter()
        output_tokens = len(output_text.split())
        prompt_tokens = len(prompt["prompt"].split())

        if first_token_time is None or output_tokens == 0:
            return RequestResult(
                request_id=request_id,
                prompt_len=prompt_tokens, output_len=0,
                ttft_ms=0, e2e_latency_ms=0, tpot_ms=0,
                itl_ms=[], output_tps=0,
                success=False, error="No tokens generated"
            )

        ttft_ms = (first_token_time - start_time) * 1000
        e2e_ms = (end_time - start_time) * 1000
        tpot_ms = (e2e_ms - ttft_ms) / max(output_tokens - 1, 1)

        itl_ms = []
        for i in range(1, len(token_timestamps)):
            itl_ms.append((token_timestamps[i] - token_timestamps[i-1]) * 1000)

        output_tps = output_tokens / ((end_time - start_time) or 1e-9)

        return RequestResult(
            request_id=request_id,
            prompt_len=prompt_tokens, output_len=output_tokens,
            ttft_ms=ttft_ms, e2e_latency_ms=e2e_ms, tpot_ms=tpot_ms,
            itl_ms=itl_ms, output_tps=output_tps, success=True
        )

    except asyncio.TimeoutError:
        return RequestResult(
            request_id=request_id,
            prompt_len=len(prompt["prompt"].split()), output_len=0,
            ttft_ms=0, e2e_latency_ms=0, tpot_ms=0,
            itl_ms=[], output_tps=0,
            success=False, error="Timeout"
        )
    except Exception as e:
        return RequestResult(
            request_id=request_id,
            prompt_len=len(prompt["prompt"].split()), output_len=0,
            ttft_ms=0, e2e_latency_ms=0, tpot_ms=0,
            itl_ms=[], output_tps=0,
            success=False, error=str(e)[:200]
        )


async def run_benchmark(
    base_url: str,
    framework: str,
    model: str,
    gpu: str,
    prompts: list,
    concurrency: int,
    num_requests: int = 300,
    warmup_requests: int = 10,
    max_tokens: int = 128,
    timeout_sec: int = 120,
) -> BenchmarkResult:
    """
    Benchmark one framework at one concurrency level.
    Warmup requests are sent first and excluded from measurement.
    """
    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    timeout = aiohttp.ClientTimeout(total=timeout_sec)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:

        # Warmup — triggers CUDA graph compilation, excluded from results
        log(f"  [WARMUP] Sending {warmup_requests} warmup requests...")
        warmup_sem = asyncio.Semaphore(min(concurrency, warmup_requests))

        async def warmup_task(idx):
            async with warmup_sem:
                await send_request(session, base_url, prompts[idx % len(prompts)],
                                   idx, model, max_tokens)

        await asyncio.gather(*[warmup_task(i) for i in range(warmup_requests)])
        log(f"  [WARMUP] Complete. Starting measurement...")

        # Measurement phase
        sem = asyncio.Semaphore(concurrency)
        results: list[RequestResult] = []

        async def benchmark_task(idx):
            async with sem:
                result = await send_request(
                    session, base_url, prompts[idx % len(prompts)],
                    idx, model, max_tokens
                )
                results.append(result)
                done = len(results)
                if done % 100 == 0:
                    log(f"  [PROGRESS] {done}/{num_requests} requests completed")

        wall_start = time.perf_counter()
        await asyncio.gather(*[benchmark_task(i) for i in range(num_requests)])
        wall_end = time.perf_counter()
        duration = wall_end - wall_start

    # Compute aggregate metrics
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    if not successful:
        log(f"  [ERROR] All {len(results)} requests failed!")
        return BenchmarkResult(
            framework=framework, model=model, gpu=gpu,
            concurrency=concurrency, num_requests=num_requests,
            duration_sec=duration, error_count=len(failed), success_rate=0.0,
        )

    ttfts = [r.ttft_ms for r in successful]
    tpots = [r.tpot_ms for r in successful]
    e2es = [r.e2e_latency_ms for r in successful]
    all_itls = [itl for r in successful for itl in r.itl_ms]

    total_output = sum(r.output_len for r in successful)
    total_input = sum(r.prompt_len for r in successful)

    result = BenchmarkResult(
        framework=framework, model=model, gpu=gpu,
        concurrency=concurrency, num_requests=num_requests,
        duration_sec=round(duration, 2),
        total_output_tokens=total_output,
        total_input_tokens=total_input,
        system_tps=round(total_output / duration, 2),
        requests_per_sec=round(len(successful) / duration, 2),

        ttft_p50=round(np.percentile(ttfts, 50), 2),
        ttft_p95=round(np.percentile(ttfts, 95), 2),
        ttft_p99=round(np.percentile(ttfts, 99), 2),
        ttft_mean=round(np.mean(ttfts), 2),

        tpot_p50=round(np.percentile(tpots, 50), 2),
        tpot_p95=round(np.percentile(tpots, 95), 2),
        tpot_p99=round(np.percentile(tpots, 99), 2),
        tpot_mean=round(np.mean(tpots), 2),

        e2e_p50=round(np.percentile(e2es, 50), 2),
        e2e_p95=round(np.percentile(e2es, 95), 2),
        e2e_p99=round(np.percentile(e2es, 99), 2),
        e2e_mean=round(np.mean(e2es), 2),

        itl_p50=round(np.percentile(all_itls, 50), 2) if all_itls else 0,
        itl_p95=round(np.percentile(all_itls, 95), 2) if all_itls else 0,
        itl_p99=round(np.percentile(all_itls, 99), 2) if all_itls else 0,
        itl_mean=round(np.mean(all_itls), 2) if all_itls else 0,

        success_rate=round(len(successful) / len(results) * 100, 1),
        error_count=len(failed),
        per_request=[asdict(r) for r in successful],
    )

    # Print summary
    log(f"\n  {'='*60}")
    log(f"  {framework.upper()} | Concurrency={concurrency} | GPU={gpu}")
    log(f"  {'='*60}")
    log(f"  Throughput:     {result.system_tps} tok/s | {result.requests_per_sec} req/s")
    log(f"  TTFT (ms):      P50={result.ttft_p50}  P95={result.ttft_p95}  P99={result.ttft_p99}")
    log(f"  TPOT (ms):      P50={result.tpot_p50}  P95={result.tpot_p95}  P99={result.tpot_p99}")
    log(f"  E2E  (ms):      P50={result.e2e_p50}  P95={result.e2e_p95}  P99={result.e2e_p99}")
    log(f"  ITL  (ms):      P50={result.itl_p50}  P95={result.itl_p95}  P99={result.itl_p99}")
    log(f"  Success:        {result.success_rate}% ({len(successful)}/{len(results)})")
    log(f"  Duration:       {result.duration_sec}s")
    log(f"  {'='*60}\n")

    return result


# ──────────────────────────────────────────────
# Sweep runner
# ──────────────────────────────────────────────

async def run_full_sweep(config: dict):
    """Run benchmark across all frameworks and concurrency levels in the config."""
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_sharegpt_dataset(
        config.get("dataset_path", "sharegpt.json"),
        num_prompts=config.get("num_prompts", 300)
    )

    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for fw_config in config["frameworks"]:
        framework = fw_config["name"]
        base_url = fw_config["base_url"]
        model = fw_config["model"]
        gpu = config["gpu"]

        log(f"\n{'#'*70}")
        log(f"# BENCHMARKING: {framework.upper()}")
        log(f"# Model: {model}")
        log(f"# Server: {base_url}")
        log(f"# GPU: {gpu}")
        log(f"{'#'*70}\n")

        # Quick health check
        try:
            async with aiohttp.ClientSession() as session:
                for endpoint in ["/health", "/v1/models", "/"]:
                    try:
                        async with session.get(f"{base_url}{endpoint}",
                                               timeout=aiohttp.ClientTimeout(total=10)) as resp:
                            if resp.status == 200:
                                log(f"  [HEALTH] {framework} is up at {base_url}")
                                break
                    except:
                        continue
                else:
                    log(f"  [WARN] Could not verify {framework} health at {base_url}")
        except Exception as e:
            log(f"  [ERROR] Cannot reach {framework} at {base_url}: {e}")
            continue

        for conc in config["concurrency_levels"]:
            log(f"\n  >>> Testing concurrency = {conc}")

            try:
                result = await run_benchmark(
                    base_url=base_url, framework=framework,
                    model=model, gpu=gpu, prompts=prompts,
                    concurrency=conc,
                    num_requests=config.get("num_requests", 300),
                    warmup_requests=config.get("warmup_requests", 10),
                    max_tokens=config.get("max_tokens", 128),
                    timeout_sec=config.get("timeout_sec", 120),
                )

                # Save per-request detail
                detail_file = results_dir / f"{framework}_c{conc}_{timestamp}_detail.json"
                with open(detail_file, "w") as f:
                    json.dump(asdict(result), f, indent=2)

                # Save aggregate (strip per-request to keep file small)
                result.per_request = []
                all_results.append(asdict(result))

                agg_file = results_dir / f"aggregate_{timestamp}.json"
                with open(agg_file, "w") as f:
                    json.dump({
                        "config": {k: v for k, v in config.items() if k != "frameworks"},
                        "frameworks": [fw["name"] for fw in config["frameworks"]],
                        "timestamp": timestamp,
                        "results": all_results
                    }, f, indent=2)

                log(f"  [SAVED] {detail_file.name}")

            except Exception as e:
                log(f"  [ERROR] {framework} @ concurrency={conc}: {e}")
                continue

            await asyncio.sleep(2)  # Cool-down between levels

        log(f"\n  [COOLDOWN] Waiting 5s before next framework...")
        await asyncio.sleep(5)

    log(f"\n{'='*70}")
    log(f"BENCHMARK COMPLETE")
    log(f"Results saved to: {results_dir}")
    log(f"Aggregate file: aggregate_{timestamp}.json")
    log(f"{'='*70}")

    return all_results


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Inference Benchmark")
    parser.add_argument("--config", type=str, required=True, help="Config JSON file")
    parser.add_argument("--frameworks", nargs="+", help="Run only these frameworks")
    parser.add_argument("--concurrency", nargs="+", type=int, help="Override concurrency levels")
    parser.add_argument("--num-requests", type=int, help="Override request count")

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    if args.frameworks:
        config["frameworks"] = [fw for fw in config["frameworks"]
                                 if fw["name"] in args.frameworks]
    if args.concurrency:
        config["concurrency_levels"] = args.concurrency
    if args.num_requests:
        config["num_requests"] = args.num_requests

    log(f"\n{'='*70}")
    log(f"LLM INFERENCE BENCHMARK SUITE")
    log(f"Config: {args.config}")
    log(f"GPU: {config['gpu']}")
    log(f"Frameworks: {[fw['name'] for fw in config['frameworks']]}")
    log(f"Concurrency levels: {config['concurrency_levels']}")
    log(f"Requests per level: {config['num_requests']}")
    log(f"Max tokens: {config.get('max_tokens', 128)}")
    log(f"{'='*70}\n")

    asyncio.run(run_full_sweep(config))