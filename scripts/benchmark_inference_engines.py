"""Benchmarks PyTorch, ONNX Runtime, and TensorRT inference latency/throughput for the DADS
dual-head model — the real numbers behind the "Inference Engine" comparison in the PhD writeup.

Run locally (CPU) to benchmark PyTorch vs ONNX Runtime — TensorRT is skipped automatically since
it needs an NVIDIA GPU. Run the same script unchanged on a RunPod GPU pod to also get the
TensorRT leg (it detects `trtexec` on PATH and includes it only when available).

Usage:
    python scripts/benchmark_inference_engines.py
    python scripts/benchmark_inference_engines.py --iterations 1000 --fp16
    python scripts/benchmark_inference_engines.py --weights data/models/current_model_weights.pt

Random weights are used if no checkpoint is found — fine for latency benchmarking, since
inference speed depends on the model's architecture/shape, not its learned values.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.backbones import DualHeadEWModel

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(ROOT_DIR, "data", "benchmark_results")


def load_model(config_path: str, weights_path: str, device: str) -> tuple:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)["model"]

    model = DualHeadEWModel(
        backbone_type=cfg["backbone_type"], input_dim=cfg["input_dim"],
        embed_dim=cfg["embedding_dim"], num_classes=cfg["num_classes"],
    )
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        print(f"Loaded trained weights from {weights_path}")
    else:
        print(f"No checkpoint at {weights_path} — benchmarking with random-init weights "
              f"(fine for latency; architecture/shape is what determines speed).")
    model.to(device).eval()
    return model, cfg


def export_onnx(model: DualHeadEWModel, input_dim: int, onnx_path: str) -> None:
    dummy = torch.randn(1, input_dim)
    torch.onnx.export(
        model.cpu(), dummy, onnx_path,
        input_names=["pdw"], output_names=["probs", "embeddings"],
        dynamic_axes={"pdw": {0: "batch"}, "probs": {0: "batch"}, "embeddings": {0: "batch"}},
        opset_version=17, dynamo=False,
    )


def summarize(latencies_ms: list) -> dict:
    arr = np.array(latencies_ms)
    return {
        "mean_ms": round(float(arr.mean()), 4),
        "p50_ms": round(float(np.percentile(arr, 50)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "p99_ms": round(float(np.percentile(arr, 99)), 4),
        "throughput_per_sec": round(1000.0 / arr.mean(), 2),
    }


def bench_pytorch(model, input_dim, device, batch_size, iterations, warmup):
    x = torch.randn(batch_size, input_dim, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()

        latencies = []
        with torch.no_grad():
            for _ in range(iterations):
                start = time.perf_counter()
                model(x)
                if device == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000)
    return summarize(latencies)


def bench_onnxruntime(onnx_path, input_dim, batch_size, iterations, warmup, use_cuda):
    import onnxruntime as ort
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    actual_provider = session.get_providers()[0]

    x = np.random.randn(batch_size, input_dim).astype(np.float32)
    inputs = {"pdw": x}
    for _ in range(warmup):
        session.run(None, inputs)

    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run(None, inputs)
        latencies.append((time.perf_counter() - start) * 1000)
    result = summarize(latencies)
    result["provider"] = actual_provider
    return result


def bench_tensorrt(onnx_path, input_dim, batch_size, iterations, warmup, fp16):
    """Builds and benchmarks a TensorRT engine via the Python API (pip-installable — no SDK
    tarball/trtexec binary needed) using PyTorch CUDA tensors directly as the I/O bindings, so
    there's no extra buffer-management dependency (pycuda) either. Requires a CUDA device."""
    try:
        import tensorrt as trt
    except ImportError:
        return None
    if not torch.cuda.is_available():
        print("TensorRT leg needs a CUDA device — none available here.")
        return None

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # TensorRT 10+ dropped implicit-batch mode entirely, so create_network() no longer takes
    # (or needs) the EXPLICIT_BATCH flag — older (8.x) bindings still require it.
    if hasattr(trt, "NetworkDefinitionCreationFlag") and hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    else:
        network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    fixed_shape = (batch_size, input_dim)
    profile.set_shape(input_tensor.name, fixed_shape, fixed_shape, fixed_shape)
    config.add_optimization_profile(profile)

    print("Building TensorRT engine (fp16={})...".format(fp16))
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("TensorRT engine build failed.")
        return None

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()
    context.set_input_shape(input_tensor.name, fixed_shape)

    io_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    output_names = [n for n in io_names if n != input_tensor.name]

    x = torch.randn(batch_size, input_dim, device="cuda")
    outputs = {}
    for name in output_names:
        shape = tuple(context.get_tensor_shape(name))
        outputs[name] = torch.empty(shape, device="cuda")

    context.set_tensor_address(input_tensor.name, x.data_ptr())
    for name in output_names:
        context.set_tensor_address(name, outputs[name].data_ptr())

    stream = torch.cuda.Stream()
    for _ in range(warmup):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
    return summarize(latencies)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=os.path.join(ROOT_DIR, "config", "config.yaml"))
    parser.add_argument("--weights", default=os.path.join(ROOT_DIR, "data", "models", "current_model_weights.pt"))
    parser.add_argument("--onnx", default=os.path.join(ROOT_DIR, "data", "models", "current_model.onnx"))
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--fp16", action="store_true", help="Use FP16 for the TensorRT engine build.")
    parser.add_argument("--skip-tensorrt", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    model, cfg = load_model(args.config, args.weights, device)
    input_dim = cfg["input_dim"]

    os.makedirs(os.path.dirname(args.onnx), exist_ok=True)
    export_onnx(model, input_dim, args.onnx)
    model.to(device)
    print(f"ONNX export: {args.onnx}\n")

    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "backbone_type": cfg["backbone_type"],
        "batch_size": args.batch_size,
        "iterations": args.iterations,
    }

    print("--- PyTorch ---")
    results["pytorch"] = bench_pytorch(model, input_dim, device, args.batch_size, args.iterations, args.warmup)
    print(results["pytorch"])

    print("\n--- ONNX Runtime ---")
    results["onnxruntime"] = bench_onnxruntime(
        args.onnx, input_dim, args.batch_size, args.iterations, args.warmup, use_cuda=(device == "cuda")
    )
    print(results["onnxruntime"])

    if args.skip_tensorrt:
        print("\n--- TensorRT: skipped (--skip-tensorrt) ---")
        results["tensorrt"] = None
    else:
        print("\n--- TensorRT ---")
        tensorrt_result = bench_tensorrt(args.onnx, input_dim, args.batch_size, args.iterations, args.warmup, args.fp16)
        if tensorrt_result is None:
            print("TensorRT unavailable or benchmark failed — this leg needs `pip install tensorrt` "
                  "and a CUDA device (e.g. a RunPod GPU pod). Skipped.")
        else:
            print(tensorrt_result)
        results["tensorrt"] = tensorrt_result

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"benchmark_{int(time.time())}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
