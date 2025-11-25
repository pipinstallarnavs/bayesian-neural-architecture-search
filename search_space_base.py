from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np

class SearchSpaceBase(ABC):
    """Base class for all search spaces"""
    
    @abstractmethod
    def enumerate(self) -> List[Dict[str, Any]]:
        """Return list of all possible architectures"""
        pass
    
    @abstractmethod
    def encode(self, arch: Dict[str, Any]) -> np.ndarray:
        """Encode architecture to vector"""
        pass
    
    @abstractmethod
    def evaluate(self, arch: Dict[str, Any]) -> float:
        """Evaluate architecture, return performance metric"""
        pass
    
    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Dimension of encoded vector"""
        pass