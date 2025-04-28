import numpy as np
import torch
import onnx
import onnx_tensorrt.backend as backend
import time
from .base_engine import InferenceEngine


class TensorRTONNXEngine(InferenceEngine):
    """TensorRT inference engine using onnx_tensorrt.backend"""
    
    def __init__(self):
        self.engine = None
    
    @property
    def name(self):
        return "TensorRT-ONNX"
    
    def load_model(self, model_path):
        """Build a TensorRT engine from the given ONNX file"""
        onnx_model = onnx.load(model_path)
        start_time = time.time()
        self.engine = backend.prepare(onnx_model, device="CUDA:0")
        engine_build_time = time.time() - start_time
        return engine_build_time
    
    def prepare_input(self, images):
        return torch.stack(images).numpy().astype(np.float32)
    
    def run(self, input_data):
        """Run inference on the input data"""
        if self.engine is None:
            raise RuntimeError("Engine not initialized. Call load_model first.")
        
        return self.engine.run(input_data)


    def get_logits_from_output(self, output):
        """
        Extract logits from onnx-tensorrt backend output.

        onnx_tensorrt.backend.prepare(...) returns a tuple/list whose first
        element is a NumPy array of shape [B, C, H, W].

        Returns a list with one [C, H, W] array per image.
        """
        # Unwrap tuple/list produced by the backend
        if isinstance(output, (list, tuple)):
            output = output[0]

        if not isinstance(output, np.ndarray):
            raise TypeError(f"TensorRTONNXEngine expected ndarray, got {type(output)}")

        return [output[i] for i in range(output.shape[0])]

