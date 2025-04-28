import os
import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
import time
from .base_engine import InferenceEngine


class TensorRTEngine(InferenceEngine):
    """TensorRT inference engine using pure TensorRT API"""
    
    def __init__(self):
        self.engine = None
        self.context = None
        self.stream = cuda.Stream()
        self.input_binding = None
        self.output_binding = None
        self.d_input = None
        self.d_output = None
        self.h_output = None
        self.output_shape = None
    
    @property
    def name(self):
        return "TensorRT"
    
    def load_model(self, model_path, workspace_size=1<<30):
        """Build a TensorRT engine from the given ONNX file"""
        
        start_time = time.time()
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        builder = trt.Builder(logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, logger)

        with open(model_path, "rb") as model_file:
            model_data = model_file.read()
        if not parser.parse(model_data):
            raise RuntimeError("Failed to parse ONNX model.")
        
        # Configure engine build
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

        # Build a serialized engine
        serialized_engine = builder.build_serialized_network(network, config)
        if not serialized_engine:
            raise RuntimeError("Failed to build a serialized TensorRT network!")
        
        # Deserialize it into an engine
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        if self.engine is None:
            raise RuntimeError("Failed to build TensorRT engine.")
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        engine_build_time = time.time() - start_time
        return engine_build_time
    
    def allocate_buffers(self, input_shape, output_shape):
        """Allocate GPU memory for input and output"""
        self.output_shape = output_shape
        
        # Calculate sizes
        input_size = int(np.prod(input_shape) * np.dtype(np.float32).itemsize)
        output_size = int(np.prod(output_shape) * np.dtype(np.float32).itemsize)
        
        # Allocate GPU memory
        self.d_input = cuda.mem_alloc(input_size)
        self.d_output = cuda.mem_alloc(output_size)
        
        # Pinned memory for outputs
        self.h_output = cuda.pagelocked_empty(shape=output_shape, dtype=np.float32)

    def prepare_input(self, images):
        return torch.stack(images).numpy().astype(np.float32)
    
    def run(self, input_data):
        """Run inference on the input data"""
        if self.d_input is None or self.d_output is None:
            batch_size, channels, height, width = input_data.shape
            self.allocate_buffers(input_data.shape, (batch_size, 19, height, width))
            
        # Copy input to GPU
        cuda.memcpy_htod_async(self.d_input, input_data, self.stream)
        
        # Run inference
        self.context.execute_async_v2(
            bindings=[int(self.d_input), int(self.d_output)], 
            stream_handle=self.stream.handle
        )
        
        # Copy output back to CPU
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        
        return self.h_output


    def get_logits_from_output(self, output):
        """
        Split the TensorRT output (NumPy array) into per-image logits.

        Args
        ----
        output : np.ndarray    # shape [B, C, H, W]

        Returns
        -------
        list[np.ndarray]       # one [C, H, W] array per image
        """
        if not isinstance(output, np.ndarray):
            raise TypeError(f"TensorRTEngine expected np.ndarray, got {type(output)}")

        # Use views – no data copy
        return [output[i] for i in range(output.shape[0])]


    # === Cache Management Methods (previously in TensorRTEngineManager) ===
    
    def save_engine(self, path):
        """
        Serialize this TensorRT engine to a plan file.
        
        Args:
            path: Path where to save the serialized engine
        """
        if self.engine is None:
            raise RuntimeError("Engine is not initialized.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Serialize the engine
        plan = self.engine.serialize()
        with open(path, "wb") as f:
            f.write(plan)
        print(f"[INFO] Successfully cached engine to {path}")
            
    @classmethod
    def load_engine(cls, path):
        """
        Load a serialized TensorRT engine from a plan file.
        
        Args:
            path: Path to the serialized engine file
            
        Returns:
            An initialized TensorRTEngine instance
        """
        logger = trt.Logger(trt.Logger.WARNING)
        # Initialize plugins for the deserializer
        trt.init_libnvinfer_plugins(logger, "")
        runtime = trt.Runtime(logger)
        
        # Read the serialized plan
        with open(path, "rb") as f:
            plan = f.read()
            
        # Deserialize into a TensorRT engine
        cuda_engine = runtime.deserialize_cuda_engine(plan)
        if cuda_engine is None:
            raise RuntimeError(f"Failed to load engine from {path}")
            
        # Create and initialize the engine class
        instance = cls()
        
        # Set the engine and context
        instance.engine = cuda_engine
        instance.context = cuda_engine.create_execution_context()
        
        return instance
    
    @classmethod
    def load_or_build(cls, model_path, cache_path, *load_args, **load_kwargs):
        """
        Load a cached TensorRT engine or build and cache a new one.
        
        Args:
            model_path: Path to the ONNX model file
            cache_path: Path where to save/load the serialized engine
            *load_args, **load_kwargs: Additional arguments for load_model
            
        Returns:
            Tuple of (engine instance, build time in seconds)
        """
        # Try to load from cache first
        if os.path.exists(cache_path):
            try:
                instance = cls.load_engine(cache_path)
                return instance, 0.0  # No build time when loading from cache
            except Exception as e:
                print(f"[WARNING] Failed to load cached engine: {e}")
                # If loading fails, fall back to building
                
        # Build a new engine
        instance = cls()
        build_time = instance.load_model(model_path, *load_args, **load_kwargs)
        
        # Cache the engine for future use
        try:
            instance.save_engine(cache_path)
        except Exception as e:
            print(f"[WARNING] Failed to cache engine: {e}")
            
        return instance, build_time
