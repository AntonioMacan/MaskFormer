import os
from pathlib import Path
import argparse
import numpy as np
import torch
from time import perf_counter
from trt.engines.tensorrt_engine import TensorRTEngine
from trt.engines.tensorrt_onnx_engine import TensorRTONNXEngine
from trt.engines.pytorch_engine import PyTorchEngine
from trt.data.utils import prepare_data


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark MaskFormer inference')
    parser.add_argument(
        '--engine',
        type=str,
        choices=['pytorch', 'tensorrt', 'tensorrt-onnx'],
        default='tensorrt', 
        help='Inference engine to use'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default="weights/maskformer_R101_bs16_90k/model_final_38c00c.pkl",
        help='Path to the model (ONNX/PyTorch)'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='datasets/cityscapes',
        help='Path to the dataset'
    )
    parser.add_argument(
        '--num_images',
        type=int,
        default=230,
        help='Number of images to benchmark'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=50,
        help='Number of warmup inferences'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=180,
        help='Number of benchmark inferences'
    )
    parser.add_argument(
        '--engine_cache_dir',
        type=str,
        default='trt/engine_cache',
        help='Directory to load/save cached engines (.plan/pickle)'
    )
    return parser.parse_args()


def compute_pixel_accuracy(pred1, pred2):
    assert pred1.shape == pred2.shape, "Shape mismatch"
    return np.mean(pred1 == pred2)


def run_benchmark(engine, data_loader, n_warmup=50, n_inference=180):
    """
    Generic benchmark function for any inference engine
    """
    loader_iter = iter(data_loader)

    # Warm-up phase
    print(f"[INFO] Running {n_warmup} warm-up inferences (not measured).")
    with torch.no_grad():
        for i in range(n_warmup):
            try:
                images, paths = next(loader_iter)
            except StopIteration:
                loader_iter = iter(data_loader)
                images, paths = next(loader_iter)

            input_data = engine.prepare_input(images)
            _ = engine.run(input_data)
    
    # Ensure CUDA operations are done
    torch.cuda.synchronize()

    print(f"[INFO] Measuring time for {n_inference} inferences...")
    times = []
    with torch.no_grad():
        for i in range(n_inference):
            try:
                images, paths = next(loader_iter)
            except StopIteration:
                loader_iter = iter(data_loader)
                images, paths = next(loader_iter)

            input_data = engine.prepare_input(images)
            start_time = perf_counter()
            _ = engine.run(input_data)
            end_time = perf_counter()
            times.append(end_time - start_time)
    
    times = np.array(times)
    total_time = times.sum()
    mean_time = times.mean()
    fps = 1.0 / mean_time

    result = {
        'total_time': total_time,
        'mean_time': mean_time,
        'fps': fps
    }

    return result


def main():
    args = parse_args()
    sizes = [(128, 256), (256, 512), (512, 1024)]  # (H, W) pairs
    results = []
    
    for size in sizes:
        H, W = size
        print(f"\n=== Benchmarking {args.engine} at resolution {H}x{W} ===")

        model_path = args.weights
        if args.engine in ['tensorrt', 'tensorrt-onnx']:
            model_path = f"trt/onnx_models/trt_model_{H}x{W}.onnx"

        if args.engine == 'pytorch':
            engine = PyTorchEngine()
            build_time = engine.load_model(model_path)
            print(f"[INFO] Built in {build_time:.2f}s")
        elif args.engine == 'tensorrt':
            cache_path = os.path.join(args.engine_cache_dir, f"{args.engine}_{H}x{W}.plan")
            engine, build_time = TensorRTEngine.load_or_build(model_path, cache_path)
            print(f"[INFO] {'Loaded from cache' if build_time == 0 else f'Built in {build_time:.2f}s'}")
        elif args.engine == 'tensorrt-onnx':
            engine = TensorRTONNXEngine()
            build_time = engine.load_model(model_path)
            print(f"[INFO] Built in {build_time:.2f}s")

        data_loader = prepare_data(
            args.dataset_path, 'val', 
            num_images=args.num_images, 
            image_size=size
        )
        
        result = run_benchmark(
            engine, data_loader, 
            n_warmup=args.warmup, 
            n_inference=args.iterations
        )

        result['size'] = size

        # === Izračun pixel-wise accuracy ako nije PyTorch ===
        if args.engine != "pytorch":
            torch_engine = PyTorchEngine()
            torch_engine.load_model(
                "weights/maskformer_R101_bs16_90k/model_final_38c00c.pkl",
                image_size=size
            )

            torch_loader = prepare_data(
                args.dataset_path, 'val',
                num_images=5,
                image_size=size
            )
            images, _ = next(iter(torch_loader))

            input_torch = torch_engine.prepare_input(images)
            out_torch = torch_engine.run(input_torch)
            preds_torch = [np.argmax(t.cpu().numpy(), axis=0) for t in out_torch]

            input_target = engine.prepare_input(images)
            out_target = engine.run(input_target)
            logits_target = engine.get_logits_from_output(out_target)
            preds_target = [np.argmax(t, axis=0) for t in logits_target]

            pixel_accs = [
                compute_pixel_accuracy(p1, p2)
                for p1, p2 in zip(preds_torch, preds_target)
            ]
            pixel_accuracy = sum(pixel_accs) / len(pixel_accs)
            result['pixel_accuracy'] = pixel_accuracy

        results.append(result)
        print("=== Done ===")

    for result in results:
        print(f"Image size: {result['size']}")
        print(f"Inference time: {result['total_time']:.2f} s")
        print(f"Mean inference time: {result['mean_time'] * 1000:.2f} ms")
        print(f"Throughput: {result['fps']:.2f} fps")
        if 'pixel_accuracy' in result:
            print(f"Pixel-wise accuracy vs PyTorch: {result['pixel_accuracy'] * 100:.2f}%")
        print()


if __name__ == "__main__":
    main()
