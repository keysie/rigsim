"""Storage size of ggml tensor types: bytes per block / weights per block, from ggml's block definitions
(ggml-common.h). Counted, not fitted."""
from __future__ import annotations

# type -> (bytes per block, weights per block)
BLOCKS = {
    "f32": (4, 1), "f16": (2, 1), "bf16": (2, 1),
    "q4_0": (18, 32),        # fp16 scale + 32 x 4 bit
    "q4_1": (20, 32),        # fp16 scale + fp16 min + 32 x 4 bit
    "q5_0": (22, 32), "q5_1": (24, 32),
    "q8_0": (34, 32),        # fp16 scale + 32 x int8
    "q2_k": (84, 256), "q3_k": (110, 256), "q4_k": (144, 256), "q5_k": (176, 256), "q6_k": (210, 256),
    "q8_k": (292, 256),
    "iq4_nl": (18, 32), "iq4_xs": (136, 256),
}


def bytes_for(type_: str, n_weights: float) -> float:
    b, w = BLOCKS[type_.lower()]
    return n_weights * b / w


def bits_per_weight(type_: str) -> float:
    b, w = BLOCKS[type_.lower()]
    return 8 * b / w
