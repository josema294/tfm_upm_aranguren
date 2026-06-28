from __future__ import annotations

from collections import deque

import numpy as np


class WindowBuffer:
    def __init__(self, window_size: int, window_step: int):
        self.window_size = window_size
        self.window_step = window_step
        self.samples: deque[dict] = deque()

    def extend(self, samples: list[dict]) -> None:
        self.samples.extend(samples)

    def pop_ready_windows(self) -> list[list[dict]]:
        windows = []
        while len(self.samples) >= self.window_size:
            window = list(self.samples)[: self.window_size]
            windows.append(window)
            for _ in range(self.window_step):
                if self.samples:
                    self.samples.popleft()
        return windows


def window_to_array(window: list[dict]) -> np.ndarray:
    return np.array(
        [[row["acc_x_g"], row["acc_y_g"], row["acc_z_g"]] for row in window],
        dtype=np.float32,
    )


def quality_report(window: list[dict], expected_size: int) -> dict:
    seqs = [int(row["seq"]) for row in window]
    timestamps = [int(row["timestamp_ms"]) for row in window]
    seq_gaps = [b - a for a, b in zip(seqs, seqs[1:])]
    lost_samples = sum(max(0, gap - 1) for gap in seq_gaps)
    timestamp_deltas = [b - a for a, b in zip(timestamps, timestamps[1:])]
    return {
        "samples_expected": expected_size,
        "samples_received": len(window),
        "seq_start": seqs[0],
        "seq_end": seqs[-1],
        "lost_samples": lost_samples,
        "max_seq_gap": max(seq_gaps, default=0),
        "timestamp_start_ms": timestamps[0],
        "timestamp_end_ms": timestamps[-1],
        "median_dt_ms": float(np.median(timestamp_deltas)) if timestamp_deltas else None,
    }
