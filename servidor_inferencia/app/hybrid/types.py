from __future__ import annotations

from typing import Protocol


class DetectorLike(Protocol):
    def predict(self, samples: list[dict], quality: dict) -> dict: ...
