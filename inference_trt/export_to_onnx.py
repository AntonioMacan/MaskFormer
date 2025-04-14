import torch
import argparse
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from mask_former import add_mask_former_config


def parse_args():
    parser = argparse.ArgumentParser(description='Export MaskFormer model to ONNX')
    parser.add_argument('--config-file', type=str,
                        default='configs/cityscapes-19/maskformer_R101_bs16_90k.yaml',
                        help='Path to the config file')
    parser.add_argument('--weights', type=str,
                        default='weights/maskformer_R101_bs16_90k/model_final_38c00c.pkl',
                        help='Path to the model weights')
    parser.add_argument('--output', type=str, 
                        help='Path to save the ONNX model')
    parser.add_argument('--height', 
                        type=int, 
                        default=512,
                        help='Input image height')
    parser.add_argument('--width', 
                        type=int, 
                        default=1024,
                        help='Input image width')
    return parser.parse_args()


def setup_cfg(args):
    # Load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    cfg.merge_from_file(args.config_file)

    # Set the specific height and width for inference
    cfg.INPUT.MIN_SIZE_TEST = args.height
    cfg.INPUT.MAX_SIZE_TEST = args.width

    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.DEVICE = "cuda"

    cfg.freeze()
    return cfg


def main():
    args = parse_args()

    # Setup configuration
    print(f"Creating model from config: {args.config_file}")
    cfg = setup_cfg(args)
    model = build_model(cfg)
    model.eval()

    # Load weights
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    print("MaskFormer model loaded successfully.")

    class MaskFormerONNXWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, batch):
            # assuming that images have shape: [batch_size, 3, H, W]
            inputs = [{"image": img} for img in batch]
            outputs = self.model(inputs)
            sem_segs = [output["sem_seg"] for output in outputs]
            return torch.stack(sem_segs)

    # Create dummy input
    input_shape = (3, args.height, args.width)
    dummy_input = torch.randn(1, *input_shape).to(cfg.MODEL.DEVICE)

    # Wrap model for ONNX export
    wrapped_model = MaskFormerONNXWrapper(model).to(cfg.MODEL.DEVICE)

    # Run a forward pass through the model for tracing
    print("Running a forward pass through the model...")
    with torch.no_grad():
        _ = wrapped_model(dummy_input)
    
    output_path = f'inference_trt/trt_model_{args.height}x{args.width}.onnx'

    # Export model to ONNX format
    print(f"Exporting model to ONNX format: {output_path}")
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    
    print(f"ONNX model has been saved as '{output_path}'")

    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified successfully!")
    except ImportError:
        print("ONNX not installed. Skipping model verification.")
    except Exception as e:
        print(f"Error verifying ONNX model: {e}")


if __name__ == "__main__":
    main()
