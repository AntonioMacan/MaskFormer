import torch
import numpy as np
from time import perf_counter
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from mask_former import add_mask_former_config
from .utils import prepare_data

def setup_cfg(image_size):
    # Load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    cfg.merge_from_file('configs/cityscapes-19/maskformer_R101_bs16_90k.yaml')

    # Set the specific height and width for inference
    cfg.INPUT.MIN_SIZE_TEST = image_size[0]
    cfg.INPUT.MAX_SIZE_TEST = image_size[1]

    cfg.MODEL.WEIGHTS = 'weights/maskformer_R101_bs16_90k/model_final_38c00c.pkl'
    cfg.MODEL.DEVICE = "cuda"

    cfg.freeze()
    return cfg

def run_benchmark(model, data_loader, n_warmup=50, n_inference=180):
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
            input_data = [{"image": image} for image in images]
            outputs = model(input_data)
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

            # Measure inference time
            start_time = perf_counter()
            input_data = [{"image": image} for image in images]
            outputs = model(input_data)
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

        cfg = setup_cfg(size)
        
        # Build model
        model = build_model(cfg)
        model.eval()
        
        # Load weights
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        result = run_benchmark(model, data_loader)
        result['size'] = size
        results.append(result)
        print("=== Done ===")

    for result in results:
        print(f"Image size: {result['size']}")
        print(f"Inference time: {result['total_time']:.2f} s")
        print(f"Mean inference time: {result['mean_time'] * 1000:.2f} ms")
        print(f"Throughput: {result['fps']:.2f} fps")
        print()

if __name__ == "__main__":
    main()