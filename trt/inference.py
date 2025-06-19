import os
import argparse
import numpy as np
from pathlib import Path
from trt.engines.tensorrt_engine import TensorRTEngine
from trt.engines.tensorrt_onnx_engine import TensorRTONNXEngine 
from trt.engines.pytorch_engine import PyTorchEngine
from trt.data.utils import prepare_data, save_segmentation_result


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with MaskFormer')
    parser.add_argument(
        "--engine",
        type=str,
        choices=["pytorch", "tensorrt", "tensorrt-onnx"],
        default="pytorch",
        help="Inference engine to use (default: pytorch)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/maskformer_R101_bs16_90k/model_final_38c00c.pkl",
        help="Path to the PyTorch model weights",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default="trt/onnx_models/trt_model_512x1024.onnx",
        help="Path to the ONNX model (for TensorRT engines)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/cityscapes",
        help="Path to the dataset root",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="trt/results",
        help="Directory where results will be written, preserving Cityscapes folder structure",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=None,
        help="Number of images to run inference on (all if None)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Input image height"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Input image width"
    )
    parser.add_argument(
    '--engine_cache_dir',
    type=str,
    default='trt/engine_cache',
    help='Directory to load/save cached engines (.plan/pickle)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    out_root = Path(args.output_dir)
    # Create output directory if it doesn't exist
    out_root.mkdir(parents=True, exist_ok=True)    
    
    # Determine which model path to use
    if args.engine == "pytorch":
        model_path = args.weights
    else:  # TensorRT engines
        model_path = args.onnx
    
    # Create engine based on choice
    if args.engine == "pytorch":
        engine = PyTorchEngine()
        build_time = engine.load_model(model_path)
        print(f"[INFO] Using {engine.name} engine")
        print(f"[INFO] Built in {build_time:.2f}s")
    elif args.engine == 'tensorrt':
        # Use caching for TensorRT engine
        cache_path = os.path.join(
            args.engine_cache_dir, 
            f"{args.engine}_{args.height}x{args.width}.plan"
        )
        engine, build_time = TensorRTEngine.load_or_build(
            model_path, 
            cache_path
        )
        print(f"[INFO] Using {engine.name} engine")
        print(f"[INFO] {'Loaded from cache' if build_time == 0 else f'Built in {build_time:.2f}s'}")
    elif args.engine == 'tensorrt-onnx':
        # Always build fresh for TensorRT-ONNX
        engine = TensorRTONNXEngine()
        build_time = engine.load_model(model_path)
        print(f"[INFO] Using {engine.name} engine")
        print(f"[INFO] Built in {build_time:.2f}s")
    
    data_loader = prepare_data(
        args.dataset_path,
        subset="val",
        num_images=args.num_images,
        batch_size=args.batch_size,
        image_size=(args.height, args.width),
    )

    dataset_root = Path(args.dataset_path)

    for images, paths, labels in data_loader:
        input_data = engine.prepare_input(images)
        output = engine.run(input_data)
        logits_batch = engine.get_logits_from_output(output)

        for logits, img_path in zip(logits_batch, paths):
            pred = np.argmax(logits, axis=0).astype(np.uint8)

            img_path = Path(img_path)
            # Compute relative path w.r.t. dataset root (keeps subset/city folders)
            rel_path = img_path.relative_to(dataset_root)

            pred_name = f"pred_{args.engine}_{args.height}x{args.width}{rel_path.suffix}"
            save_path = out_root / rel_path.parent / rel_path.stem / pred_name
            save_path.parent.mkdir(parents=True, exist_ok=True)

            save_segmentation_result(pred, save_path)
            print(f"[INFO] Saved result: {save_path}")


if __name__ == "__main__":
    main()
