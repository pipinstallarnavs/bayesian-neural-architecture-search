import numpy as np
from typing import Dict, Any, List, Tuple
from search_space_base import SearchSpaceBase
from encoder import OneHotEdgeOpEncoder

try:
    from nats_bench import create
    NATS_AVAILABLE = True
except ImportError:
    NATS_AVAILABLE = False

class NASBench201Space(SearchSpaceBase):
    """
    NAS-Bench-201 with Correct Topological Sorting and Graph Encoding.
    """
    
    OPS = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    
    # CORRECT Topological Order: Sorted by Target Node (1, 2, 3)
    EDGES = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
    
    def __init__(self, api_path: str, dataset: str = 'cifar10', use_12epoch: bool = True):
        if not NATS_AVAILABLE:
            raise ImportError("Install: pip install nats-bench")
        
        # Load API (fast_mode=True loads the pre-computed index)
        self.api = create(api_path, 'tss', fast_mode=True, verbose=False)
        self.dataset = dataset
        self.use_12epoch = use_12epoch
        self.encoder = OneHotEdgeOpEncoder(self.EDGES, self.OPS)
        self.op_map = {op: i for i, op in enumerate(self.OPS)}
        
        # Pre-compute Line Graph Adjacency for GNN
        # Nodes in Line Graph = The 6 Edges of the NAS Cell
        self.num_edges = len(self.EDGES)
        self.line_graph_adj = np.zeros((self.num_edges, self.num_edges), dtype=np.float32)
        for i, (src_i, tgt_i) in enumerate(self.EDGES):
            for j, (src_j, tgt_j) in enumerate(self.EDGES):
                # If Edge A's target is Edge B's source, data flows A -> B
                if tgt_i == src_j:
                    self.line_graph_adj[i, j] = 1.0
    
    def enumerate(self) -> List[Dict[str, Any]]:
        archs = []
        for idx in range(len(self.api)):
            arch_str = self.api.query_by_index(idx).arch_str
            edge_ops = self._parse_arch_str(arch_str)
            archs.append({'index': idx, 'arch_str': arch_str, 'edge_ops': edge_ops})
        return archs
    
    def _parse_arch_str(self, arch_str: str) -> Dict[Tuple[int, int], str]:
        nodes = arch_str.strip('|').split('|+|')
        edge_ops = {}
        for node_id, node_str in enumerate(nodes, start=1):
            ops = node_str.split('|')
            for op_str in ops:
                if '~' in op_str:
                    op_name, from_node = op_str.split('~')
                    edge_ops[(int(from_node), node_id)] = op_name
        return edge_ops
    
    def encode(self, arch: Dict[str, Any]) -> np.ndarray:
        """Standard Flat One-Hot Encoding (for MLP/CNN)."""
        return self.encoder.encode(arch)

    def encode_graph(self, arch: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Graph Encoding for GNN.
        Returns:
            op_indices: (6,) array of integer op indices
            adjacency: (6, 6) adjacency matrix
        """
        ops = []
        for edge in self.EDGES:
            op_name = arch['edge_ops'][edge]
            ops.append(self.op_map[op_name])
        return np.array(ops, dtype=np.int64), self.line_graph_adj
    
    def evaluate(self, arch: Dict[str, Any]) -> float:
        info = self.api.get_more_info(
            arch['index'], self.dataset,
            hp='12' if self.use_12epoch else '200', is_random=False
        )
        return float(info['valid-accuracy'])
    
    @property
    def input_dim(self) -> int:
        return self.encoder.input_dim()