import numpy as np
from typing import Dict, Any, List, Tuple

class OneHotEdgeOpEncoder:
    """
    Simple vector encoder:
    - For each possible edge (i->j), one-hot the chosen op (e.g., conv3x3, conv1x1, skip, zero).
    - Concatenate all edge-op one-hots into a single vector.
    """
    def __init__(self, all_edges: List[Tuple[int,int]], op_names: List[str]):
        self.edges = all_edges
        self.op_names = op_names
        self.op_to_idx = {op:i for i,op in enumerate(op_names)}
        self.dim_per_edge = len(op_names)
        self.dim = len(all_edges) * self.dim_per_edge

    def encode(self, arch_spec: Dict[str, Any]) -> np.ndarray:
        """
        arch_spec: {'edge_ops': { (i,j): 'conv3x3', ... } }
        """
        vec = np.zeros(self.dim, dtype=np.float32)
        for e_idx, e in enumerate(self.edges):
            op = arch_spec["edge_ops"][e]
            k = self.op_to_idx[op]
            start = e_idx * self.dim_per_edge
            vec[start + k] = 1.0
        return vec

    def input_dim(self) -> int:
        return self.dim
