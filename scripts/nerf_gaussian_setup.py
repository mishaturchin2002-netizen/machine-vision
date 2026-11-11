"""
nerf_gaussian_setup.py — scaffolding for NeRF / Gaussian Splatting training
===========================================================================

!!!!!  DO NOT RUN THIS ON THIS LAPTOP  !!!!!

This machine has an Intel Core Ultra 7 + Intel Arc *integrated* graphics and no
CUDA. nerfstudio / gsplat / tiny-cuda-nn require an NVIDIA GPU with CUDA >= 11.8
(12.1 recommended) and roughly 12-24 GB of VRAM depending on the scene. The
install alone is several gigabytes, and on Windows the gsplat rasteriser has to
compile CUDA extensions with MSVC — it will not build here.

This file is therefore *scaffolding only*: it prints the commands you would run
on an NVIDIA machine, and refuses to install or train unless you pass --yes
AND a working CUDA device is detected.

--------------------------------------------------------------------------------
WHAT YOU NEED ON THE NVIDIA MACHINE
--------------------------------------------------------------------------------
  - NVIDIA driver >= 535, CUDA toolkit 12.1
  - Python 3.10 or 3.11 (3.12 is not supported by nerfstudio yet)
  - 16+ GB RAM, 12+ GB VRAM
  - Your photos in ./images/ (20-200 shots, overlapping, well lit)

--------------------------------------------------------------------------------
QUICK START (on the NVIDIA box, from this project folder)
--------------------------------------------------------------------------------
  # 0. sanity check
  python nerf_gaussian_setup.py check

  # 1. create an isolated environment (never mix with the cv venv)
  python -m venv C:\\nerf-venv
  C:\\nerf-venv\\Scripts\\activate

  # 2a. install nerfstudio (simpler, includes its own viewer)
  python nerf_gaussian_setup.py install --method nerfstudio --yes

  # 2b. or install gsplat (faster 3D Gaussian Splatting, more manual)
  python nerf_gaussian_setup.py install --method gsplat --yes

  # 3. turn the photos into a dataset (COLMAP: camera poses + sparse points)
  python nerf_gaussian_setup.py prepare --images ./images --out ./data/scene

  # 4. train
  python nerf_gaussian_setup.py train --method nerfstudio --data ./data/scene
  python nerf_gaussian_setup.py train --method gsplat     --data ./data/scene

  # 5. render a fly-through / export the point cloud
  python nerf_gaussian_setup.py export --method nerfstudio --data ./data/scene

--------------------------------------------------------------------------------
NOTES / GOTCHAS
--------------------------------------------------------------------------------
  * Photograph the object from a full circle, ~20 deg between shots, fixed
    exposure and focus, no motion blur. NeRF hates blurry photos.
  * COLMAP on Windows needs the CUDA build for speed; the CPU build works but
    takes ~10x longer. `--no-gpu` below forces the CPU path.
  * gsplat training of a small object: ~10-20 min on an RTX 3060 at 30k steps.
  * If you get "CUDA out of memory", reduce --max-res and the number of images.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

IMAGES_DIR = Path("./images")
DATA_DIR = Path("./data/scene")
OUTPUT_DIR = Path("./output")
MIN_CUDA = (11, 8)
RECOMMENDED_CUDA = (12, 1)


# --------------------------------------------------------------------------
# Environment checks (safe: only reads state, installs nothing)
# --------------------------------------------------------------------------
def check_environment(verbose: bool = True) -> bool:
    """Report whether this machine can realistically run NeRF/GSplat."""
    problems: list[str] = []
    notes: list[str] = []

    # --- python version -------------------------------------------------
    major, minor = sys.version_info[:2]
    notes.append(f"python            : {major}.{minor} ({sys.executable})")
    if (major, minor) not in ((3, 10), (3, 11)):
        problems.append("nerfstudio supports Python 3.10/3.11 only")

    # --- NVIDIA driver --------------------------------------------------
    smi = shutil.which("nvidia-smi")
    if smi is None:
        problems.append("nvidia-smi not found -> no NVIDIA driver installed here")
    else:
        try:
            out = subprocess.run([smi, "--query-gpu=name,memory.total,driver_version",
                                  "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=15)
            notes.append(f"gpu               : {out.stdout.strip() or 'unknown'}")
        except Exception as exc:                                  # noqa: BLE001
            problems.append(f"nvidia-smi failed: {exc}")

    # --- torch + CUDA ---------------------------------------------------
    try:
        import torch
        notes.append(f"torch             : {torch.__version__} "
                     f"(cuda build: {torch.version.cuda})")
        if not torch.cuda.is_available():
            problems.append("torch.cuda.is_available() is False -> training would run on CPU "
                            "(impractical)")
        else:
            notes.append(f"cuda device       : {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            notes.append(f"vram              : {vram:.1f} GB")
            if vram < 6:
                problems.append(f"only {vram:.1f} GB VRAM - expect out-of-memory")
    except ImportError:
        problems.append("torch is not installed in this environment")

    # --- optional tools -------------------------------------------------
    for tool, why in (("colmap", "COLMAP: needed to compute camera poses"),
                      ("ffmpeg", "ffmpeg: needed to extract frames from video")):
        notes.append(f"{tool:18s}: {'found' if shutil.which(tool) else 'MISSING'} ({why})")

    if verbose:
        print("=" * 70)
        print("ENVIRONMENT CHECK")
        print("=" * 70)
        for line in notes:
            print("  " + line)
        print("-" * 70)
        if problems:
            print("BLOCKERS:")
            for p in problems:
                print(f"  [x] {p}")
            print("\n=> This machine cannot train NeRF/Gaussian Splatting as is.")
            print("   Use an NVIDIA machine with CUDA >= 11.8.")
        else:
            print("OK - this machine looks capable. Use --yes to actually install/train.")
        print("=" * 70)

    return not problems


# --------------------------------------------------------------------------
# Install (guarded, never implicit)
# --------------------------------------------------------------------------
NERFSTUDIO_STEPS = [
    # torch with CUDA 12.1 wheels; do this BEFORE nerfstudio so pip does not
    # pull a CPU-only torch and silently "succeed"
    "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121",
    "pip install nerfstudio",
    # tiny-cuda-nn is optional but speeds up Nerfacto a lot
    "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch",
]

GSPLAT_STEPS = [
    "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121",
    "pip install gsplat",
    "pip install nerfview",
    # COLMAP (Windows): use the CUDA build from https://github.com/colmap/colmap/releases
    "# verify with:  colmap -h",
]


def install(method: str, assume_yes: bool) -> int:
    steps = NERFSTUDIO_STEPS if method == "nerfstudio" else GSPLAT_STEPS
    print(f"Install plan for '{method}' ({len(steps)} steps):\n")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}")
    print("\nThis is a multi-gigabyte download and needs an NVIDIA GPU with CUDA.")
    print("It is deliberately NOT executed automatically.\n")
    if not assume_yes:
        print("Refusing to install without --yes (and a detected CUDA device).")
        return 1
    if not check_environment(verbose=True):
        print("\nRefusing: environment check failed.")
        return 1
    for i, step in enumerate(steps, 1):
        if step.strip().startswith("#"):
            continue
        print(f"\n[{i}/{len(steps)}] {step}", flush=True)
        parts = step.split()
        result = subprocess.run(parts, shell=False)
        if result.returncode != 0:
            print(f"[!] step failed with exit code {result.returncode}")
            return result.returncode
    print("\nDone. Activate this environment before training.")
    return 0


# --------------------------------------------------------------------------
# Dataset preparation (COLMAP)
# --------------------------------------------------------------------------
def prepare(images: Path, out: Path, no_gpu: bool, dry_run: bool) -> int:
    if not images.exists():
        print(f"[!] image folder not found: {images}")
        print("    put 20-200 overlapping photos of the object/scene there")
        return 1
    photos = sorted(p for p in images.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    print(f"[i] {len(photos)} image(s) in {images}")
    if len(photos) < 12:
        print("[!] fewer than 12 images - COLMAP will struggle to recover poses")

    if dry_run and shutil.which("ns-process-data") is None:
        print("[dry-run] would run:")
    cmd = ["ns-process-data", "images",
           "--data", str(images),
           "--output-dir", str(out),
           "--num-downscales", "2"]
    if no_gpu:
        cmd += ["--no-gpu"]          # CPU SIFT: same result, ~10x slower
    print("  " + " ".join(cmd))
    if dry_run:
        return 0
    if shutil.which("ns-process-data") is None:
        print("[!] ns-process-data not found - install nerfstudio first")
        return 1
    return subprocess.run(cmd, shell=False).returncode


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train(method: str, data: Path, max_steps: int, dry_run: bool) -> int:
    if not data.exists():
        print(f"[!] dataset not found: {data} - run 'prepare' first")
        return 1

    if method == "nerfstudio":
        cmd = ["ns-train", "nerfacto",
               "--data", str(data),
               "--max-num-iterations", str(max_steps),
               "--output-dir", str(OUTPUT_DIR)]
        notes = "Nerfacto: ~30-60 min on an RTX 3060 for 30k iters."
    else:
        # gsplat ships examples/simple_trainer.py; this mirrors its CLI
        cmd = ["python", "-m", "gsplat.examples.simple_trainer",
               "default",
               "--data_dir", str(data),
               "--max_steps", str(max_steps),
               "--result_dir", str(OUTPUT_DIR)]
        notes = "gsplat: 3D Gaussian Splatting, usually faster to converge than Nerfacto."

    print(f"[i] {notes}")
    print("  " + " ".join(cmd))
    if dry_run:
        print("\n[dry-run] nothing executed. Drop --dry-run on the NVIDIA machine.")
        return 0
    if not check_environment(verbose=False):
        print("[!] needs NVIDIA CUDA - aborting")
        return 1
    return subprocess.run(cmd, shell=False).returncode


# --------------------------------------------------------------------------
# Export / render
# --------------------------------------------------------------------------
def export(method: str, data: Path, dry_run: bool) -> int:
    if method == "nerfstudio":
        cmd = ["ns-export", "pointcloud",
               "--load-config", str(OUTPUT_DIR / "nerfacto" / "<run>" / "config.yml"),
               "--output-dir", str(OUTPUT_DIR / "pointcloud.ply")]
        extra = ["ns-render", "interpolate",
                 "--load-config", str(OUTPUT_DIR / "nerfacto" / "<run>" / "config.yml"),
                 "--output-path", str(OUTPUT_DIR / "flythrough.mp4")]
    else:
        cmd = ["python", "-m", "gsplat.examples.simple_trainer", "default",
               "--data_dir", str(data), "--export_ply", str(OUTPUT_DIR / "splat.ply")]
        extra = []

    print("export:")
    print("  " + " ".join(cmd))
    for e in extra:
        print("  " + " ".join(e))
    if dry_run:
        return 0
    return subprocess.run(cmd, shell=False).returncode


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="NeRF / Gaussian Splatting scaffolding. Nothing is installed or trained "
                    "unless --yes is passed AND CUDA is detected.")
    sub = ap.add_subparsers(dest="cmd")

    p_check = sub.add_parser("check", help="report whether this machine can run NeRF/GSplat")

    p_install = sub.add_parser("install", help="install nerfstudio or gsplat (guarded)")
    p_install.add_argument("--method", choices=("nerfstudio", "gsplat"), default="nerfstudio")
    p_install.add_argument("--yes", action="store_true", help="actually run the install")

    p_prep = sub.add_parser("prepare", help="COLMAP: images -> camera poses")
    p_prep.add_argument("--images", type=Path, default=IMAGES_DIR)
    p_prep.add_argument("--out", type=Path, default=DATA_DIR)
    p_prep.add_argument("--no-gpu", action="store_true", help="CPU COLMAP (slow)")
    p_prep.add_argument("--dry-run", action="store_true", default=True)

    p_train = sub.add_parser("train", help="start training")
    p_train.add_argument("--method", choices=("nerfstudio", "gsplat"), default="nerfstudio")
    p_train.add_argument("--data", type=Path, default=DATA_DIR)
    p_train.add_argument("--max-steps", type=int, default=30000)
    p_train.add_argument("--dry-run", action="store_true", default=True)

    p_export = sub.add_parser("export", help="export point cloud / render video")
    p_export.add_argument("--method", choices=("nerfstudio", "gsplat"), default="nerfstudio")
    p_export.add_argument("--data", type=Path, default=DATA_DIR)
    p_export.add_argument("--dry-run", action="store_true", default=True)

    args = ap.parse_args()
    if args.cmd is None:
        ap.print_help()
        print("\nStart with:  python nerf_gaussian_setup.py check")
        return 0
    if args.cmd == "check":
        return 0 if check_environment(verbose=True) else 1
    if args.cmd == "install":
        return install(args.method, args.yes)
    if args.cmd == "prepare":
        return prepare(args.images, args.out, args.no_gpu, args.dry_run)
    if args.cmd == "train":
        return train(args.method, args.data, args.max_steps, args.dry_run)
    if args.cmd == "export":
        return export(args.method, args.data, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
