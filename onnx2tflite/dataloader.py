import os
import logging
import numpy as np

LOG = logging.getLogger("Quantization Dataloader:")


class RandomLoader:
    """Generate random calibration data (low accuracy, for quick testing only)."""
    def __init__(self, input_shapes: list):
        self.shapes = input_shapes
        LOG.warning("Using random calibration data — accuracy will be degraded!")

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i > 50:
            raise StopIteration()
        self._i += 1
        return [np.random.randn(*s).astype(np.float32) for s in self.shapes]


class NpyLoader:
    """Load preprocessed calibration data from .npy files.

    Each .npy file should contain a batch of representative input data.
    Multiple files can be provided for multiple model inputs.
    """
    def __init__(self, npy_paths: list[str], input_shapes: list):
        assert len(npy_paths) == len(input_shapes), \
            f"Expected {len(input_shapes)} .npy files (one per model input), got {len(npy_paths)}"
        self.data = []
        for path, shape in zip(npy_paths, input_shapes):
            assert os.path.exists(path), f"Calibration file not found: {path}"
            arr = np.load(path)
            assert arr.shape[1:] == tuple(shape[1:]), \
                f"Shape mismatch in {path}: data has {arr.shape[1:]}, model expects {shape[1:]}"
            self.data.append(arr)
        self._num_samples = min(len(d) for d in self.data)
        LOG.info(f"{self._num_samples} calibration samples loaded from {len(npy_paths)} .npy file(s)")

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i >= self._num_samples:
            raise StopIteration()
        batch = [d[self._i:self._i + 1].astype(np.float32) for d in self.data]
        self._i += 1
        return batch
