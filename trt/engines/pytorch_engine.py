import torch
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from mask_former import add_mask_former_config
from .base_engine import InferenceEngine

class PyTorchEngine(InferenceEngine):
    """PyTorch inference engine for MaskFormer model"""
    
    def __init__(self):
        self.model = None
        self.cfg = None
    
    @property
    def name(self):
        return "PyTorch"
    
    def setup_cfg(self, config_file, image_size, weights_path):
        """Setup configuration for the model"""
        # Load config from file and command-line arguments
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_mask_former_config(cfg)
        cfg.merge_from_file(config_file)

        # Set the specific height and width for inference
        H, W = image_size
        cfg.INPUT.MIN_SIZE_TEST = min(H, W)
        cfg.INPUT.MAX_SIZE_TEST = max(H, W)

        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.DEVICE = "cuda"

        cfg.freeze()
        return cfg
    
    def load_model(self, model_path, config_file=None, image_size=(512, 1024)):
        """
        Load a PyTorch MaskFormer model
        
        Args:
            model_path: Path to model weights
            config_file: Path to config file (if None, use default)
            image_size: Image size for inference (H, W)
        """
        if config_file is None:
            config_file = 'configs/cityscapes-19/maskformer_R101_bs16_90k.yaml'
        
        # Setup configuration
        self.cfg = self.setup_cfg(config_file, image_size, model_path)
        
        # Build model
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        self.model = build_model(self.cfg)
        self.model.eval()
        
        # Load weights
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)
        
        end_time.record()
        torch.cuda.synchronize()
        load_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        
        return load_time
    
    def prepare_input(self, images):
        return [{"image": image} for image in images]

    def run(self, input_data):
        """
        Run inference with the PyTorch model
        
        Args:
            input_data: List of dictionaries with 'image' key
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call load_model first.")
        
        with torch.no_grad():
            outputs = self.model(input_data)
        
        # Extract semantic segmentation outputs
        sem_segs = [output["sem_seg"] for output in outputs]
        return torch.stack(sem_segs)
    

    def get_logits_from_output(self, output):
        """
        Convert Detectron2 output to a list of NumPy logits.

        Args
        ----
        output : torch.Tensor  # shape [B, C, H, W]

        Returns
        -------
        list[np.ndarray]       # length == batch-size, each [C, H, W]
        """
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"PyTorchEngine expected torch.Tensor, got {type(output)}")

        # Split the batch tensor into per-image NumPy arrays (no extra copying).
        return [tensor.cpu().numpy() for tensor in output]
