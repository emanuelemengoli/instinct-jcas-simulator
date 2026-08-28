"""Deterministic named random-number streams."""

from __future__ import annotations

import hashlib
import numpy as np


class RNGManager:
    """Create independent deterministic RNG streams from one master seed.

    Stream seeds are derived from a stable cryptographic digest of the stream
    name, so results do not depend on wall-clock time, thread scheduling, or
    the order in which unrelated streams are first requested.
    """

    def __init__(self, master_seed: int):
        self.master_seed = int(master_seed)
        self._cache: dict[str, np.random.Generator] = {}

    def generator(self, name: str) -> np.random.Generator:
        if name not in self._cache:
            digest = hashlib.blake2b(name.encode("utf-8"), digest_size=16).digest()
            words = np.frombuffer(digest, dtype=np.uint32).astype(np.uint64).tolist()
            seed_sequence = np.random.SeedSequence([self.master_seed, *words])
            self._cache[name] = np.random.default_rng(seed_sequence)
        return self._cache[name]

    def fresh_generator(self, name: str) -> np.random.Generator:
        """Return a fresh generator at the beginning of a named stream."""
        digest = hashlib.blake2b(name.encode("utf-8"), digest_size=16).digest()
        words = np.frombuffer(digest, dtype=np.uint32).astype(np.uint64).tolist()
        return np.random.default_rng(np.random.SeedSequence([self.master_seed, *words]))
