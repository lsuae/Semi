from __future__ import annotations

from pathlib import Path

import numpy as np


def _print_shape(path: Path, tag: str) -> None:
    if not path.exists():
        return
    arr = np.load(path, mmap_mode="r")
    print(tag, str(path).replace("\\", "/"), arr.shape, arr.dtype)


def main() -> None:
    for ds in ["cifar100", "food101", "eurosat", "stl10"]:
        _print_shape(Path("Data") / ds / f"features_{ds}.npy", f"{ds}:features")
        _print_shape(Path("Data") / ds / f"pseudo_label_confidences_{ds}.npy", f"{ds}:pseudo_conf")
        _print_shape(Path("Data") / ds / f"cm_{ds}.npy", f"{ds}:cm")


if __name__ == "__main__":
    main()
