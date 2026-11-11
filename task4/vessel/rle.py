"""One-based, row-major RLE of a boolean vessel mask."""

import numpy as np


def encode(mask: np.ndarray) -> str:
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("RLE input must be a two-dimensional boolean vessel mask")
    flat = mask.ravel(order="C").astype(np.int8)
    transitions = np.flatnonzero(np.diff(np.concatenate(([0], flat, [0]))))
    starts, lengths = transitions[::2] + 1, transitions[1::2] - transitions[::2]
    return " ".join(str(int(value)) for pair in zip(starts, lengths) for value in pair)


def decode(encoded: str, shape: tuple[int, int]) -> np.ndarray:
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("RLE shape must contain two positive dimensions")
    tokens = [int(value) for value in encoded.split()]
    if len(tokens) % 2:
        raise ValueError("RLE must contain start/length pairs")
    flat = np.zeros(int(np.prod(shape)), dtype=bool)
    previous_end = 0
    for start, length in zip(tokens[::2], tokens[1::2]):
        if start <= 0 or length <= 0 or start <= previous_end or start - 1 + length > flat.size:
            raise ValueError("RLE contains an invalid, unsorted or overlapping run")
        flat[start - 1 : start - 1 + length] = True
        previous_end = start - 1 + length
    return flat.reshape(shape, order="C")
