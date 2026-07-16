#!/usr/bin/env python3
"""
install_pytorch_linux.py

A defensive, paranoid installer/verifier for PyTorch on Linux.

Assumes every step CAN and WILL go wrong at least once:
  - wrong Python / architecture (x86_64 vs aarch64 vs 32-bit)
  - running on an NVIDIA Jetson (Tegra) board, where PyPI wheels have NO CUDA support
  - pip missing, broken, or ancient
  - no write permission / read-only filesystem / disk full
  - old glibc (manylinux wheel incompatibility) common on older distros
  - no internet, corporate proxy, SSL cert failure
  - stale/conflicting torch installs (apt + pip + conda mixed)
  - GPU present but driver too old / CUDA mismatch / nvidia-smi missing
  - missing build tools needed for source builds if wheels don't match
  - pip cache corruption
  - OMP/MKL duplicate runtime crash
  - not running inside a virtual environment (system Python pollution)
  - locale/encoding issues (rare but real on minimal containers)

Usage:
    python3 install_pytorch_linux.py
    python3 install_pytorch_linux.py --cpu-only
    python3 install_pytorch_linux.py --log-file setup.log

Exit codes:
    0 = success (torch importable and functional)
    1 = failed, see report/log
    2 = succeeded with warnings (torch works, but degraded, e.g. CPU-only fallback)
"""

import sys
import os
import re
import time
import shutil
import logging
import platform
import subprocess
from pathlib import Path

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

def build_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("pytorch_setup")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    try:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError as e:
        logger.warning(f"Could not open log file '{log_file}': {e}. Continuing console-only.")

    return logger


REPORT = {"errors": [], "warnings": [], "info": []}


def note(kind, msg, logger):
    REPORT[kind].append(msg)
    if kind == "errors":
        logger.error(msg)
    elif kind == "warnings":
        logger.warning(msg)
    else:
        logger.info(msg)


# --------------------------------------------------------------------------
# Generic safe command runner
# --------------------------------------------------------------------------

def run(cmd, logger, retries=1, retry_delay=3, timeout=600, **kw):
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            logger.debug(f"Running: {' '.join(cmd)} (attempt {attempt}/{retries})")
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, **kw,
            )
            if result.returncode == 0:
                return True, result.stdout, result.stderr
            last_err = result.stderr.strip() or result.stdout.strip()
            logger.warning(f"Command failed (exit {result.returncode}): {last_err[:400]}")
        except FileNotFoundError as e:
            last_err = f"Executable not found: {e}"
            break
        except subprocess.TimeoutExpired:
            last_err = f"Command timed out after {timeout}s"
        except OSError as e:
            last_err = f"OS-level failure launching process: {e}"
        if attempt < retries:
            time.sleep(retry_delay)
    return False, "", last_err


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def check_platform(logger):
    if platform.system() != "Linux":
        note("warnings", f"This script targets Linux; detected '{platform.system()}'. "
                          "Continuing anyway, but Linux-specific fixes may not apply.", logger)
    logger.info(f"OS: {platform.platform()}")
    logger.info(f"Architecture: {platform.machine()}")
    return True


def check_jetson(logger):
    """Jetson/Tegra boards need NVIDIA's own PyTorch builds, not PyPI wheels."""
    tegra_release = Path("/etc/nv_tegra_release")
    if tegra_release.exists():
        try:
            info = tegra_release.read_text().strip()
        except OSError:
            info = "(unreadable)"
        note("info", f"NVIDIA Jetson / Tegra device detected ({info}). Standard 'pip install torch' "
                      "from PyPI has no CUDA support here -- will use the Jetson AI Lab pip index instead, "
                      "which hosts NVIDIA-compatible aarch64+CUDA wheels matched to your JetPack version.",
             logger)
        return True
    return False


def get_l4t_major(logger):
    """Parse /etc/nv_tegra_release for the L4T release number, which maps to a JetPack major version
    (R35.x -> JetPack 5.x, R36.x -> JetPack 6.x)."""
    try:
        text = Path("/etc/nv_tegra_release").read_text()
        match = re.search(r"R(\d+)", text)
        if match:
            l4t_major = int(match.group(1))
            jp_major = {35: 5, 36: 6, 34: 5, 32: 4}.get(l4t_major)
            if jp_major:
                logger.info(f"Detected L4T R{l4t_major} -> JetPack {jp_major}.x")
                return jp_major
            note("warnings", f"Detected L4T R{l4t_major} but no known JetPack mapping for it; "
                              "the Jetson AI Lab index layout may differ. Check "
                              "https://pypi.jetson-ai-lab.dev/ manually.", logger)
    except OSError as e:
        note("warnings", f"Could not read /etc/nv_tegra_release: {e}", logger)
    return None


def get_installed_cuda_tag(logger):
    """Returns a tag like 'cu126' from the CUDA toolkit that ships with JetPack, or None."""
    version_json = Path("/usr/local/cuda/version.json")
    if version_json.exists():
        try:
            import json
            data = json.loads(version_json.read_text())
            ver = data.get("cuda", {}).get("version")
            if ver:
                major, minor = (int(x) for x in ver.split(".")[:2])
                tag = f"cu{major}{minor}"
                logger.info(f"CUDA toolkit version (from version.json): {ver} -> {tag}")
                return tag
        except Exception as e:
            note("warnings", f"Could not parse {version_json}: {e}", logger)

    ok, out, err = run(["nvcc", "--version"], logger, retries=1, timeout=15)
    if ok:
        match = re.search(r"release (\d+)\.(\d+)", out)
        if match:
            major, minor = int(match.group(1)), int(match.group(2))
            tag = f"cu{major}{minor}"
            logger.info(f"CUDA toolkit version (from nvcc): {major}.{minor} -> {tag}")
            return tag

    note("warnings", "Could not determine the installed CUDA toolkit version (checked "
                      "/usr/local/cuda/version.json and `nvcc --version`). CUDA toolkit may not be on PATH; "
                      "try `export PATH=/usr/local/cuda/bin:$PATH` or check `sudo apt list --installed "
                      "| grep cuda`.", logger)
    return None


def install_pytorch_jetson(jp_major, cuda_tag, logger):
    """Install PyTorch/torchvision for Jetson using the Jetson AI Lab community pip index,
    which hosts NVIDIA-compatible aarch64 wheels pre-matched to JetPack/CUDA versions."""
    if not jp_major or not cuda_tag:
        note("errors", "Could not determine JetPack major version and/or CUDA toolkit version "
                        "automatically, so a matching Jetson wheel can't be selected safely. "
                        "Manually check your JetPack version (`sudo apt-cache show nvidia-jetpack`) "
                        "and CUDA version (`nvcc --version`), then browse "
                        "https://pypi.jetson-ai-lab.dev/ for the matching jp<major>/cu<xx> index, "
                        "or follow https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/ "
                        "for a direct wheel URL for your exact JetPack release.", logger)
        return False

    index_url = f"https://pypi.jetson-ai-lab.dev/jp{jp_major}/{cuda_tag}"
    logger.info(f"Attempting Jetson-matched install from: {index_url}")

    cmd = [sys.executable, "-m", "pip", "install",
           "--index-url", index_url,
           "--extra-index-url", "https://pypi.org/simple/",
           "torch", "torchvision"]

    ok, out, err = run(cmd, logger, retries=2, retry_delay=6, timeout=1800)
    if ok:
        return True

    ok2, out2, err2 = run(cmd + ["--no-cache-dir"], logger, retries=1, retry_delay=6, timeout=1800)
    if ok2:
        return True

    note("errors", f"Install from Jetson AI Lab index ({index_url}) failed: {err2 or err}. "
                    "That index may not have a build for this exact JetPack/CUDA combo yet. Next steps: "
                    "(1) browse https://pypi.jetson-ai-lab.dev/ directly to see what's actually published "
                    "for your jp/cu path, (2) check the official NVIDIA redistributables at "
                    "https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/ for a "
                    "direct .whl download matching your JetPack release, or (3) use a prebuilt container "
                    "from https://github.com/dusty-nv/jetson-containers (l4t-pytorch image) instead of a "
                    "bare pip install.", logger)
    return False


def check_python(logger):
    v = sys.version_info
    is_64 = sys.maxsize > 2**32
    logger.info(f"Python: {sys.version.split()[0]} ({'64-bit' if is_64 else '32-bit'}, {platform.machine()})")
    if not is_64:
        note("errors", "32-bit Python detected. PyTorch has no 32-bit Linux wheels. Install a 64-bit "
                        "Python build and re-run.", logger)
        return False
    if v < (3, 9) or v >= (3, 14):
        note("warnings", f"Python {v.major}.{v.minor} may not have a published PyTorch wheel yet. "
                          "3.9-3.13 is the safest range; check pytorch.org for current support.", logger)
    return True


def check_glibc(logger):
    try:
        ok, out, err = run(["ldd", "--version"], logger, retries=1, timeout=10)
        if ok:
            first_line = out.splitlines()[0] if out else ""
            match = re.search(r"(\d+\.\d+)\s*$", first_line)
            logger.info(f"glibc: {first_line.strip()}")
            if match:
                major, minor = (int(x) for x in match.group(1).split("."))
                if (major, minor) < (2, 28):
                    note("warnings", f"glibc {major}.{minor} is quite old. Modern PyTorch manylinux wheels "
                                      "generally need glibc >= 2.28 (roughly Ubuntu 18.04+/CentOS 8+). "
                                      "If pip install fails with 'not a supported wheel on this platform' "
                                      "or a mysterious segfault on import, this is a likely cause -- consider "
                                      "upgrading the distro or using a container with a newer base image.", logger)
        else:
            note("warnings", f"Could not determine glibc version ({err[:150]}). Skipping this check.", logger)
    except Exception as e:
        note("warnings", f"glibc check failed unexpectedly: {e}", logger)
    return True


def check_write_permissions(logger):
    target = Path.cwd() / ".pytorch_setup_write_test.tmp"
    try:
        target.write_text("test")
        target.unlink()
        logger.info("Write permission OK in current directory.")
        return True
    except PermissionError:
        note("errors", "No write permission in the current directory. If this is a read-only mount, "
                        "a container filesystem layer, or a root-owned folder, 'cd' into your home "
                        "directory or run with appropriate permissions (avoid blanket sudo pip installs; "
                        "prefer a virtual environment you own instead).", logger)
        return False
    except OSError as e:
        note("errors", f"Unexpected filesystem error testing write permission: {e}", logger)
        return False


def check_disk_space(logger, min_gb=8):
    try:
        total, used, free = shutil.disk_usage(str(Path.cwd()))
        free_gb = free / (1024**3)
        logger.info(f"Free disk space: {free_gb:.1f} GB")
        if free_gb < min_gb:
            note("errors", f"Only {free_gb:.1f} GB free; PyTorch + CUDA runtime + pip cache can need "
                            f"{min_gb}+ GB (Jetson devices with small eMMC/SD storage hit this a lot). "
                            "Free up space, e.g. `sudo apt clean`, clear ~/.cache/pip, or use external storage.",
                 logger)
            return False
    except OSError as e:
        note("warnings", f"Could not determine free disk space: {e}. Proceeding without this check.", logger)
    return True


def check_pip(logger):
    ok, out, err = run([sys.executable, "-m", "pip", "--version"], logger, retries=1)
    if not ok:
        note("errors", f"pip is not usable via '{sys.executable} -m pip'. Error: {err}. "
                        "Try: python3 -m ensurepip --upgrade, or "
                        "sudo apt install python3-pip.", logger)
        return False
    logger.info(f"pip: {out.strip()}")

    ok, out, err = run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                        logger, retries=2, retry_delay=5)
    if not ok:
        note("warnings", f"Could not upgrade pip automatically ({err[:200]}). An old pip can fail to "
                          "resolve platform-specific wheels (especially on aarch64). Continuing with "
                          "existing pip version.", logger)
    return True


def check_venv(logger):
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    running_as_root = os.geteuid() == 0 if hasattr(os, "geteuid") else False
    if not in_venv:
        note("warnings", "Not running inside a virtual environment. This install will affect your "
                          "system/global Python, which can conflict with distro-managed packages "
                          "(some distros even block this with 'externally-managed-environment'). "
                          "Recommended: python3 -m venv venv && source venv/bin/activate", logger)
    else:
        logger.info(f"Running inside virtual environment: {sys.prefix}")
    if running_as_root and not in_venv:
        note("warnings", "Running as root outside a virtual environment. Avoid 'sudo pip install' for "
                          "Python packages -- it can corrupt system Python tooling. Use a venv instead.", logger)
    return True


def check_internet_and_proxy(logger):
    test_urls = ["https://pypi.org", "https://download.pytorch.org"]
    for url in test_urls:
        ok, out, err = run(
            [sys.executable, "-c",
             f"import urllib.request; urllib.request.urlopen('{url}', timeout=8)"],
            logger, retries=2, retry_delay=3,
        )
        if not ok:
            if "CERTIFICATE_VERIFY_FAILED" in err:
                note("warnings", f"SSL certificate verification failed reaching {url}. Common behind "
                                  "corporate/campus proxies or if 'ca-certificates' is outdated "
                                  "(try: sudo apt install --reinstall ca-certificates). If needed, pip "
                                  "can be retried with --trusted-host pypi.org --trusted-host "
                                  "files.pythonhosted.org --trusted-host download.pytorch.org.", logger)
            elif "timed out" in err.lower():
                note("warnings", f"Timed out reaching {url}. Check network/VPN/proxy. If behind a proxy, "
                                  "set http_proxy / https_proxy environment variables.", logger)
            else:
                note("warnings", f"Could not reach {url}: {err[:200]}", logger)
        else:
            logger.info(f"Reachable: {url}")
    return True


def detect_existing_torch(logger):
    ok, out, err = run([sys.executable, "-c", "import torch; print(torch.__version__)"],
                        logger, retries=1, timeout=30)
    if ok:
        note("warnings", f"Existing torch install detected (version {out.strip()}). Mixing apt, pip, and "
                          "conda torch installs is a common source of import errors and ABI mismatches. "
                          "Will attempt a clean pip uninstall first (won't touch apt/conda-managed copies).",
             logger)
        run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"],
            logger, retries=1, timeout=120)
        return True
    return False


def detect_gpu(logger):
    """Returns a suggested CUDA tag ('cu121', 'cu124', etc.) or None for CPU-only."""
    ok, out, err = run(["nvidia-smi"], logger, retries=1, timeout=15)
    if not ok:
        note("info", "No NVIDIA GPU detected (nvidia-smi not found or failed). Installing CPU-only "
                      "PyTorch. If you do have an NVIDIA GPU, install/verify the proprietary driver "
                      "(e.g. `sudo ubuntu-drivers autoinstall` on Ubuntu, or your distro's equivalent) "
                      "and make sure the kernel module actually loaded (`lsmod | grep nvidia`).", logger)
        return None

    logger.info("nvidia-smi output detected; GPU present.")
    match = re.search(r"CUDA Version:\s*([\d.]+)", out)
    if not match:
        note("warnings", "GPU detected but couldn't parse CUDA driver version from nvidia-smi output. "
                          "Falling back to a conservative CUDA build (cu121).", logger)
        return "cu121"

    driver_cuda = match.group(1)
    logger.info(f"Driver supports CUDA up to: {driver_cuda}")
    major_minor = tuple(int(x) for x in driver_cuda.split(".")[:2])

    if major_minor >= (12, 4):
        tag = "cu124"
    elif major_minor >= (12, 1):
        tag = "cu121"
    elif major_minor >= (11, 8):
        tag = "cu118"
    else:
        note("warnings", f"GPU driver only supports CUDA {driver_cuda}, older than PyTorch's minimum "
                          "supported build (CUDA 11.8). Falling back to CPU-only PyTorch. Update your "
                          "NVIDIA driver to use GPU acceleration.", logger)
        return None
    return tag


# --------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------

def install_pytorch(cuda_tag, logger, force_cpu=False):
    if force_cpu:
        cuda_tag = None

    packages = ["torch", "torchvision", "torchaudio"]
    if cuda_tag:
        index_url = f"https://download.pytorch.org/whl/{cuda_tag}"
        logger.info(f"Installing GPU build ({cuda_tag}) from {index_url}")
    else:
        index_url = "https://download.pytorch.org/whl/cpu"
        logger.info("Installing CPU-only build.")

    base_cmd = [sys.executable, "-m", "pip", "install"] + packages + ["--index-url", index_url]

    ok, out, err = run(base_cmd, logger, retries=2, retry_delay=6, timeout=1800)
    if ok:
        return True, cuda_tag

    combined_err = err.lower()

    if "cache" in combined_err or "corrupt" in combined_err or not ok:
        note("warnings", "Initial install failed; retrying with --no-cache-dir in case of a corrupted "
                          "pip cache.", logger)
        ok, out, err = run(base_cmd + ["--no-cache-dir"], logger, retries=2, retry_delay=6, timeout=1800)
        if ok:
            return True, cuda_tag

    if "externally-managed-environment" in combined_err:
        note("warnings", "pip refused to install into the system Python (PEP 668 'externally-managed "
                          "environment', common on Debian/Ubuntu 23.04+). Retrying inside a fresh venv "
                          "would be the correct fix; as a one-off workaround, retrying with "
                          "--break-system-packages (not recommended long-term).", logger)
        ok, out, err = run(base_cmd + ["--break-system-packages"], logger, retries=1, retry_delay=5, timeout=1800)
        if ok:
            return True, cuda_tag

    if "certificate" in combined_err or "ssl" in combined_err:
        note("warnings", "SSL error during install; retrying with --trusted-host flags.", logger)
        trusted = ["--trusted-host", "pypi.org", "--trusted-host", "files.pythonhosted.org",
                   "--trusted-host", "download.pytorch.org"]
        ok, out, err = run(base_cmd + trusted, logger, retries=2, retry_delay=6, timeout=1800)
        if ok:
            return True, cuda_tag

    if cuda_tag is not None:
        note("warnings", f"GPU build ({cuda_tag}) install failed after retries: {err[:300]}. Falling back "
                          "to CPU-only build so you have a working PyTorch at all.", logger)
        cpu_cmd = [sys.executable, "-m", "pip", "install"] + packages + \
                  ["--index-url", "https://download.pytorch.org/whl/cpu", "--no-cache-dir"]
        ok, out, err = run(cpu_cmd, logger, retries=2, retry_delay=6, timeout=1800)
        if ok:
            return True, None

    note("errors", f"PyTorch installation failed after all fallback attempts. Last error: {err[:500]}", logger)
    return False, cuda_tag


# --------------------------------------------------------------------------
# Verify
# --------------------------------------------------------------------------

def verify_install(logger, expect_cuda):
    test_script = (
        "import os\n"
        "os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')\n"
        "import torch\n"
        "print('TORCH_VERSION', torch.__version__)\n"
        "print('CUDA_AVAILABLE', torch.cuda.is_available())\n"
        "x = torch.rand(3, 3)\n"
        "y = x @ x\n"
        "print('TENSOR_OK', tuple(y.shape))\n"
        "if torch.cuda.is_available():\n"
        "    g = torch.rand(3, 3, device='cuda')\n"
        "    print('GPU_TENSOR_OK', tuple((g @ g).shape))\n"
    )
    ok, out, err = run([sys.executable, "-c", test_script], logger, retries=1, timeout=60)

    if not ok:
        if "GLIBC" in err:
            note("errors", "Import failed due to a glibc version mismatch. Your distro's glibc is too "
                            "old for this wheel. Either upgrade the distro, use a newer container base "
                            "image, or install an older PyTorch release built against an older manylinux "
                            "tag.", logger)
        elif "libcudart" in err.lower() or "libcublas" in err.lower():
            note("errors", "A CUDA shared library failed to load. Check `nvidia-smi` still works, that "
                            "no conflicting CUDA toolkit is on LD_LIBRARY_PATH, and that the installed "
                            "torch CUDA build matches a driver version that supports it.", logger)
        elif "OMP" in err or "libiomp" in err.lower() or "libgomp" in err.lower():
            note("errors", "OpenMP runtime conflict. Try setting KMP_DUPLICATE_LIB_OK=TRUE, or check for "
                            "duplicate OpenMP libraries from a mixed conda+pip environment.", logger)
        elif "No module named 'torch'" in err:
            note("errors", "torch still not importable after install claimed success -- likely installed "
                            "into a different Python/environment than the one running this script.", logger)
        else:
            note("errors", f"Verification failed: {err[:500]}", logger)
        return False, False

    logger.info(out.strip())
    cuda_actually_available = "CUDA_AVAILABLE True" in out
    if expect_cuda and not cuda_actually_available:
        note("warnings", "GPU build was installed but torch.cuda.is_available() returned False. Check "
                          "driver/CUDA build compatibility, and that the GPU isn't already claimed by "
                          "another process/container without proper GPU passthrough.", logger)
    return True, cuda_actually_available


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Defensive PyTorch installer for Linux.")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU-only install, skip GPU detection.")
    parser.add_argument("--log-file", default="pytorch_setup.log", help="Path to write detailed log.")
    args = parser.parse_args()

    logger = build_logger(args.log_file)
    logger.info("=" * 70)
    logger.info("PyTorch-on-Linux defensive installer starting")
    logger.info("=" * 70)

    fatal = False

    check_platform(logger)
    is_jetson = check_jetson(logger)

    if not check_python(logger):
        fatal = True

    check_glibc(logger)

    if not check_write_permissions(logger):
        fatal = True

    if not check_disk_space(logger):
        fatal = True

    if fatal:
        print_report(logger)
        sys.exit(1)

    if not check_pip(logger):
        print_report(logger)
        sys.exit(1)

    check_venv(logger)
    check_internet_and_proxy(logger)
    detect_existing_torch(logger)

    if is_jetson and not args.cpu_only:
        jp_major = get_l4t_major(logger)
        cuda_tag = get_installed_cuda_tag(logger)
        installed = install_pytorch_jetson(jp_major, cuda_tag, logger)
        if not installed:
            note("warnings", "Jetson-matched GPU install failed. Falling back to CPU-only PyTorch from "
                              "PyPI so you at least have something to work with while you sort out the "
                              "GPU build manually (see the error above for exact next steps).", logger)
            installed, _ = install_pytorch(None, logger, force_cpu=True)
            final_cuda_tag = None
        else:
            final_cuda_tag = cuda_tag
        if not installed:
            print_report(logger)
            sys.exit(1)
    else:
        cuda_tag = None if args.cpu_only else detect_gpu(logger)
        installed, final_cuda_tag = install_pytorch(cuda_tag, logger, force_cpu=args.cpu_only)
        if not installed:
            print_report(logger)
            sys.exit(1)

    verified, cuda_working = verify_install(logger, expect_cuda=bool(final_cuda_tag))
    print_report(logger)

    if not verified:
        sys.exit(1)
    if final_cuda_tag and not cuda_working:
        logger.info("RESULT: PyTorch installed and functional on CPU; GPU not confirmed working.")
        sys.exit(2)
    if is_jetson and not final_cuda_tag:
        logger.info("RESULT: CPU-only PyTorch installed. For GPU acceleration on this Jetson device, "
                     "follow the guidance above to get NVIDIA's Jetson-specific build instead.")
        sys.exit(2)

    logger.info("RESULT: PyTorch installed and fully verified"
                f"{' with CUDA' if cuda_working else ' (CPU-only)'}.")
    sys.exit(0)


def print_report(logger):
    logger.info("-" * 70)
    logger.info("SUMMARY REPORT")
    if REPORT["errors"]:
        logger.info(f"{len(REPORT['errors'])} error(s):")
        for e in REPORT["errors"]:
            logger.info(f"  [ERROR] {e}")
    if REPORT["warnings"]:
        logger.info(f"{len(REPORT['warnings'])} warning(s):")
        for w in REPORT["warnings"]:
            logger.info(f"  [WARN]  {w}")
    if not REPORT["errors"] and not REPORT["warnings"]:
        logger.info("No issues detected.")
    logger.info("-" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"UNEXPECTED FAILURE in installer itself: {e}")
        sys.exit(1)
