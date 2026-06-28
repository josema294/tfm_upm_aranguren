from __future__ import annotations

import torch


def main() -> int:
    print(f"torch={torch.__version__}")
    print(f"hip={getattr(torch.version, 'hip', None)}")
    print(f"cuda_api_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device_count={torch.cuda.device_count()}")
        print(f"device_0={torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: PyTorch no detecta GPU ROCm dentro del contenedor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
