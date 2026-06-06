import torch
import random

class DataSource:
    """Base class for data generators that yield batches."""
    def __init__(self, dim, batch_len, silent=True):
        self.dim = dim
        self.batch_len = batch_len
        self.silent = silent

    def next(self):
        return [self._generate_one() for _ in range(self.batch_len)]

    def _generate_one(self):
        raise NotImplementedError


def str_to_bitlist(text, encoding="utf-8"):
    """Convert string to list of -1/1 bits."""
    raw = text.encode(encoding)
    bits = []
    for byte in raw:
        for i in range(8):
            bits.append(1 if (byte >> i) & 1 else -1)
    return bits


def bitlist_to_str(bits, encoding="utf-8"):
    """Convert list of -1/1 bits back to string."""
    bytes_list = []
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            if i+j < len(bits):
                byte |= (1 if bits[i+j] == 1 else 0) << j
        bytes_list.append(byte)
    return bytes(bytes_list).decode(encoding, errors="ignore")