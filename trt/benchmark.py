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
            # Get next image (with cycling if needed)
            try:
                images, paths = next(loader_iter)
            except StopIteration:
                loader_iter = iter(data_loader)
                images, paths = next(loader_iter)
            
            input_data = engine.prepare_input(images)
            
            # Measure inference time
            start_time = perf_counter()
            _ = engine.run(input_data)
            end_time = perf_counter()
            times.append(end_time - start_time)
    
    # Calculate statistics
    times = np.array(times)
    total_time= times.sum()
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
    
    # Test with different resolutions
    sizes = [(128, 256), (256, 512), (512, 1024)]  # (H, W) pairs
    results = []
    
    for size in sizes:
        H, W = size
        print(f"\n=== Benchmarking {args.engine} at resolution {H}x{W} ===")

        # Set up paths
        model_path = args.weights
        if args.engine in ['tensorrt', 'tensorrt-onnx']:
            model_path = f"trt/onnx_models/trt_model_{H}x{W}.onnx"
        
        # Create engine based on choice
        if args.engine == 'pytorch':
            engine = PyTorchEngine()
            build_time = engine.load_model(model_path)
            print(f"[INFO] Built in {build_time:.2f}s")
        elif args.engine == 'tensorrt':
            # For TensorRT, use caching
            cache_path = os.path.join(args.engine_cache_dir, f"{args.engine}_{H}x{W}.plan")
            engine, build_time = TensorRTEngine.load_or_build(
                model_path, 
                cache_path
            )
            print(f"[INFO] {'Loaded from cache' if build_time == 0 else f'Built in {build_time:.2f}s'}")
        elif args.engine == 'tensorrt-onnx':
            # For TensorRT-ONNX, always build fresh (no caching)
            engine = TensorRTONNXEngine()
            build_time = engine.load_model(model_path)
            print(f"[INFO] Built in {build_time:.2f}s")

        # Prepare data loader
        data_loader = prepare_data(
            args.dataset_path, 'val', 
            num_images=args.num_images, 
            image_size=size
        )
        
        # Run benchmark
        result = run_benchmark(
            engine, data_loader, 
            n_warmup=args.warmup, 
            n_inference=args.iterations
        )
        
        result['size'] = size
        results.append(result)
        print("=== Done ===")
    
    # Print results
    for result in results:
        print(f"Image size: {result['size']}")
        print(f"Inference time: {result['total_time']:.2f} s")
        print(f"Mean inference time: {result['mean_time'] * 1000:.2f} ms")
        print(f"Throughput: {result['fps']:.2f} fps")
        print()

if __name__ == "__main__":
    main()
