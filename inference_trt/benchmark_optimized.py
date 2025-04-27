from time import perf_counter
import numpy as np
import torch
from .utils import prepare_data
from .inference_onnx_tensorrt import build_engine_onnx

def run_benchmark(engine, data_loader, n_warmup=50, n_inference=180):
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

            input_data = torch.stack(images).numpy().astype(np.float32)
            
            _ = engine.run(input_data)
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
            
            # Prepare input data
            input_data = torch.stack(images).numpy().astype(np.float32)
            
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
    sizes = [(128, 256), (256, 512), (512, 1024)]  # (H, W) pairs
    results = []
    for size in sizes:
        H, W = size
        print(f"\n=== Benchmarking at resolution {H}x{W} ===")
        data_loader = prepare_data('datasets/cityscapes', 'val', 
                              num_images=230, 
                              image_size=size)
        onnx_path = f"inference_trt/trt_model_{H}x{W}.onnx"
        engine, _ = build_engine_onnx(onnx_path)
        result = run_benchmark(engine, data_loader)
        result['size'] = size
        results.append(result)
        del engine
        print("=== Done ===")

    for result in results:
        print(f"Image size: {result['size']}")
        print(f"Inference time: {result['total_time']:.2f} s")
        print(f"Mean inference time: {result['mean_time'] * 1000:.2f} ms")
        print(f"Throughput: {result['fps']:.2f} fps")
        print()

if __name__ == "__main__":
    main()