from abc import ABC, abstractmethod

class InferenceEngine(ABC):
    """Base abstract class for all inference engines"""
    
    @abstractmethod
    def load_model(self, model_path):
        """Load a model from the given path"""
        pass
    
    @abstractmethod
    def run(self, input_data):
        """Run inference on the input data"""
        pass
    
    @property
    @abstractmethod
    def name(self):
        """Return the name of the engine"""
        pass
