import os
import shutil
from pathlib import Path

from PIL import Image

LEVIR = Path("/scratch/gilbreth/rangarav/ChangeViT/LEVIR")
PATCH_SIZE = 256

for split in ["Val", "Test"]:
    for subdir in ["A", "B", "label"]:
        src = LEVIR / split / subdir
        tmp = LEVIR / f"{split}_tmp" / subdir
        tmp.mkdir(parents=True, exist_ok=True)

        files = sorted([f for f in os.listdir(src) if f.endswith(".png")])
        total_patches = 0
        for fname in files:
            img = Image.open(src / fname)
            w, h = img.size
            name = fname.replace(".png", "")
            idx = 0
            for y in range(0, h, PATCH_SIZE):
                for x in range(0, w, PATCH_SIZE):
                    patch = img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
                    patch.save(tmp / f"{name}_{idx}.png")
                    idx += 1
            total_patches += idx
        print(f"  {split}/{subdir}: {len(files)} images -> {total_patches} patches")

    for subdir in ["A", "B", "label"]:
        shutil.rmtree(LEVIR / split / subdir)
        os.rename(LEVIR / f"{split}_tmp" / subdir, LEVIR / split / subdir)
    shutil.rmtree(LEVIR / f"{split}_tmp")
