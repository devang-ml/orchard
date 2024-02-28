# model_args
# This class defines the model argument used by the netowrks
from dataclasses import dataclass

@dataclass
class ModelArgs:
    CONFIGS = {
        "CodeLlama-7b-Python-hf": dict(block_size=16384, vocab_size=32000, n_layer=32, dim = 4096, rope_base=1000000),
        "7B": dict(n_layer=32, n_head=32, dim=4096),
        "13B": dict(n_layer=40, n_head=40, dim=5120),
        "30B": dict(n_layer=60, n_head=52, dim=6656),
        "34B": dict(n_layer=48, n_head=64, dim=8192, vocab_size=32000, n_local_heads=8, intermediate_size=22016, rope_base=1000000), # CodeLlama-34B-Python-hf
        "70B": dict(n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672),
    }

    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5

    n_local_layer: int = -1
    n_local_head: int = -1
    local_dim: int = -1
    local_intermediate_size: int = -1
    local_head_dim: int = -1

    rank = 0
    world_size = 1

    def __post_init__(self):
        if self.n_local_layer == -1:
            self.n_local_layer = self.n_layer
        if self.n_local_head == -1:
            self.n_local_head = self.n_head
        if self.local_dim == -1:
            self.local_dim = self.dim
        if self.local_head_dim == -1:
            self.local_head_dim = self.head_dim
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = ModelArgs.find_multiple(n_hidden, 256)
        if self.local_intermediate_size == -1:
            self.local_intermediate_size = self.intermediate_size
        self.head_dim = self.dim // self.n_head

    @staticmethod
    def find_multiple(n: int, k: int) -> int:
        return n if n % k == 0 else n + k - (n % k)

    @classmethod
    def from_name(cls, name: str):
        if name in ModelArgs.CONFIGS:
            return cls(**ModelArgs.CONFIGS[name])
        # fuzzy search
        config = [config for config in ModelArgs.CONFIGS if config in str(name).upper() or config in str(name)]
        assert len(config) == 1, name
        return cls(**ModelArgs.CONFIGS[config[0]])
