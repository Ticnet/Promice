"""
procedural.py — Procedural CI/CD log generation engine.

Generates unique, semantically-equivalent failure logs per episode so that
agents cannot overfit to static strings.  Every generated log preserves the
required error-type tokens (e.g. "ModuleNotFoundError", "version conflict",
"ABI mismatch") while varying package names, versions, timestamps, file
paths, and injecting optional noise lines.

Usage:
    from env.procedural import generate_log

    rng = random.Random(42)
    log = generate_log("tier_1", rng, noise=True)
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Sequence

# ---------------------------------------------------------------------------
# Package / version pools
# ---------------------------------------------------------------------------

_PYTHON_PACKAGES: list[dict] = [
    {"name": "scikit-learn",  "import": "sklearn",      "submodule": "ensemble",       "class": "RandomForestClassifier"},
    {"name": "scipy",         "import": "scipy",        "submodule": "optimize",       "class": "minimize"},
    {"name": "transformers",  "import": "transformers", "submodule": "models",         "class": "AutoModel"},
    {"name": "lightgbm",      "import": "lightgbm",     "submodule": "",               "class": "LGBMClassifier"},
    {"name": "xgboost",       "import": "xgboost",      "submodule": "",               "class": "XGBClassifier"},
    {"name": "flask",         "import": "flask",        "submodule": "",               "class": "Flask"},
    {"name": "fastapi",       "import": "fastapi",      "submodule": "",               "class": "FastAPI"},
    {"name": "celery",        "import": "celery",       "submodule": "",               "class": "Celery"},
    {"name": "pillow",        "import": "PIL",          "submodule": "Image",          "class": "open"},
    {"name": "matplotlib",    "import": "matplotlib",   "submodule": "pyplot",         "class": "figure"},
]

_VERSION_TRIPLES: list[str] = [
    "1.2.0", "1.3.0", "1.4.1", "2.0.0", "2.1.0", "2.2.3",
    "3.0.0", "3.1.2", "0.9.1", "0.11.0", "4.0.0", "1.0.3",
]

_CACHE_PACKAGES: list[dict] = [
    {"name": "torch",       "cached": "2.0.0+cpu", "required": "2.1.0"},
    {"name": "tensorflow",  "cached": "2.12.0",    "required": "2.14.0"},
    {"name": "numpy",       "cached": "1.24.3",    "required": "1.26.0"},
    {"name": "scipy",       "cached": "1.10.1",    "required": "1.12.0"},
    {"name": "pandas",      "cached": "1.5.3",     "required": "2.1.0"},
    {"name": "jax",         "cached": "0.4.8",     "required": "0.4.20"},
    {"name": "onnxruntime", "cached": "1.14.0",    "required": "1.16.0"},
]

_GCC_PAIRS: list[dict] = [
    {"old_ver": "9",  "new_ver": "11", "old_abi": "old ABI", "new_abi": "new ABI"},
    {"old_ver": "7",  "new_ver": "12", "old_abi": "old ABI", "new_abi": "new ABI"},
    {"old_ver": "8",  "new_ver": "11", "old_abi": "old ABI", "new_abi": "new ABI"},
    {"old_ver": "9",  "new_ver": "13", "old_abi": "old ABI", "new_abi": "new ABI"},
    {"old_ver": "10", "new_ver": "14", "old_abi": "old ABI", "new_abi": "new ABI"},
]

_NATIVE_LIBS: list[dict] = [
    {"lib": "libboost_python39.so", "symbol": "std::__cxx11::basic_string", "dso": "libstdc++.so.6"},
    {"lib": "libopencv_core.so",    "symbol": "cv::Mat::deallocate",        "dso": "libstdc++.so.6"},
    {"lib": "libtorch_cpu.so",      "symbol": "c10::Error::Error",          "dso": "libstdc++.so.6"},
    {"lib": "libgrpc++.so.1",       "symbol": "grpc::Channel::Channel",     "dso": "libstdc++.so.6"},
    {"lib": "libprotobuf.so.32",    "symbol": "google::protobuf::Arena",    "dso": "libstdc++.so.6"},
]

_COMMIT_HASHES: list[str] = [
    "abc123", "f4e8d91", "c0ffee1", "deadbf2", "b4dc0d3",
    "a1b2c3d", "7e5t1ng", "p4tch42", "f1x3d99", "h0tf1x7",
]

_TEST_FILES: list[str] = [
    "tests/test_model.py", "tests/test_pipeline.py", "tests/test_api.py",
    "tests/test_utils.py", "tests/test_train.py", "tests/test_inference.py",
    "tests/test_data.py", "tests/test_config.py",
]

_EXTRA_PACKAGES: list[tuple[str, str]] = [
    ("numpy", "1.24.3"), ("pandas", "2.0.1"), ("requests", "2.31.0"),
    ("pyyaml", "6.0.1"), ("tqdm", "4.65.0"), ("click", "8.1.7"),
    ("httpx", "0.24.1"), ("aiohttp", "3.8.5"), ("boto3", "1.28.0"),
]

# Noise lines: irrelevant warnings that sometimes appear in real CI logs.
_NOISE_LINES: list[str] = [
    "WARNING: pip is configured with locations that require TLS/SSL, however the ssl module in Python is not available.",
    "DEPRECATION: Python 3.8 reached EOL. Support will be removed in a future release.",
    "WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None)) after connection broken by 'SSLError'",
    "Note: switching to 'detached HEAD' state.",
    "warning: LF will be replaced by CRLF in requirements.txt.",
    "INFO: pip is looking at multiple versions of setuptools to determine which version is compatible.",
    "WARNING: Discarding cached wheels for cryptography - hash mismatch.",
    "NOTICE: A new release of pip is available: 23.1.2 -> 24.0",
    "WARNING: The scripts pip3 and pip3.11 are installed in '/usr/local/bin' which is not on PATH.",
    "DEBUG: Fetching index from https://pypi.org/simple/ ...",
    "WARNING: Running pip as the 'root' user can result in broken permissions.",
    "INFO: Building wheel for pycparser (setup.py): started",
]

# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def _random_timestamp(rng: random.Random, base: datetime | None = None) -> datetime:
    """Generate a plausible UTC timestamp."""
    if base is None:
        base = datetime(2026, rng.randint(1, 12), rng.randint(1, 28),
                        rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59),
                        tzinfo=timezone.utc)
    return base


def _ts(base: datetime, offset_secs: int) -> str:
    """Format timestamp with offset."""
    t = base + timedelta(seconds=offset_secs)
    return t.strftime("[%Y-%m-%dT%H:%M:%SZ]")


# ---------------------------------------------------------------------------
# Log generators per tier
# ---------------------------------------------------------------------------

def _generate_tier1_log(rng: random.Random, inject_noise: bool = False) -> str:
    """Generate a missing-dependency log with randomized package details."""
    pkg = rng.choice(_PYTHON_PACKAGES)
    pkg_version = rng.choice(_VERSION_TRIPLES)
    extras = rng.sample(_EXTRA_PACKAGES, k=min(3, len(_EXTRA_PACKAGES)))
    test_file = rng.choice(_TEST_FILES)
    base_ts = _random_timestamp(rng)

    import_path = pkg["import"]
    if pkg["submodule"]:
        import_path = f"{pkg['import']}.{pkg['submodule']}"
    from_clause = f"from {import_path} import {pkg['class']}"

    lines = [
        f"{_ts(base_ts, 0)} [build] Pipeline started — stage: install_packages",
        f"{_ts(base_ts, 1)} [build] Running: pip install -r requirements.txt",
    ]
    for name, ver in extras:
        lines.append(f"{_ts(base_ts, 2)} [build] Collecting {name}=={ver}")

    lines.extend([
        f"{_ts(base_ts, 3)} [build] ERROR: Could not find a version that satisfies the requirement {pkg['name']}=={pkg_version}",
        f"{_ts(base_ts, 3)} [build] ERROR: No matching distribution found for {pkg['name']}=={pkg_version}",
        f"{_ts(base_ts, 3)} [build] ---",
    ])

    if inject_noise:
        lines.append(f"{_ts(base_ts, 4)} [build] {rng.choice(_NOISE_LINES)}")

    lines.extend([
        f"{_ts(base_ts, 4)} [test]  Running: python -m pytest {test_file.rsplit('/', 1)[0]}/",
        f"{_ts(base_ts, 5)} [test]  Traceback (most recent call last):",
        f"{_ts(base_ts, 5)} [test]    File \"{test_file}\", line {rng.randint(1, 50)}, in <module>",
        f"{_ts(base_ts, 5)} [test]      {from_clause}",
        f"{_ts(base_ts, 5)} [test]  ModuleNotFoundError: No module named '{pkg['import']}'",
        f"{_ts(base_ts, 5)} [test]  ERROR: failed to collect tests - ModuleNotFoundError",
        f"{_ts(base_ts, 5)} [build] EXIT CODE: 1",
        f"{_ts(base_ts, 6)} [build] Pipeline FAILED at stage: install_packages",
    ])

    return "\n".join(lines) + "\n"


def _generate_tier2_log(rng: random.Random, inject_noise: bool = False) -> str:
    """Generate a cache/version conflict log with randomized packages."""
    cpkg = rng.choice(_CACHE_PACKAGES)
    cache_date = f"2026-{rng.randint(1, 3):02d}-{rng.randint(1, 28):02d}"
    base_ts = _random_timestamp(rng)

    lines = [
        f"{_ts(base_ts, 0)} [build]  Pipeline started — stage: dependency_resolution",
        f"{_ts(base_ts, 1)} [build]  Running: pip install --upgrade pip",
        f"{_ts(base_ts, 2)} [build]  Running: pip install -r requirements.txt",
        f"{_ts(base_ts, 3)} [resolver] ERROR: pip's dependency resolver encountered a version conflict.",
        f"{_ts(base_ts, 3)} [resolver]   Current environment: {cpkg['name']}=={cpkg['cached']} (cached)",
        f"{_ts(base_ts, 3)} [resolver]   Requirement: {cpkg['name']}=={cpkg['required']}",
        f"{_ts(base_ts, 4)} [cache]   Stale cache detected at ~/.cache/pip/wheels/{cpkg['name']}-{cpkg['cached']}*.whl",
        f"{_ts(base_ts, 4)} [cache]   Cache timestamp: {cache_date} — packages may be outdated",
    ]

    if inject_noise:
        lines.append(f"{_ts(base_ts, 5)} [build]  {rng.choice(_NOISE_LINES)}")

    lines.extend([
        f"{_ts(base_ts, 5)} [resolver] ERROR: Cannot install {cpkg['name']}=={cpkg['required']} because of conflicting cached version {cpkg['cached']}",
        f"{_ts(base_ts, 5)} [resolver] HINT: Clear pip cache and pin the correct version in requirements.txt",
        f"{_ts(base_ts, 6)} [build]  Running: python -c \"import {cpkg['name']}; print({cpkg['name']}.__version__)\"",
        f"{_ts(base_ts, 6)} [build]  {cpkg['cached']}",
        f"{_ts(base_ts, 6)} [build]  AssertionError: expected {cpkg['name']}>={cpkg['required']}, got {cpkg['cached']}",
        f"{_ts(base_ts, 7)} [build]  EXIT CODE: 1",
        f"{_ts(base_ts, 7)} [build]  Pipeline FAILED at stage: dependency_resolution",
    ])

    return "\n".join(lines) + "\n"


def _generate_tier3_log(rng: random.Random, inject_noise: bool = False) -> str:
    """Generate an ABI mismatch log with randomized details."""
    gcc = rng.choice(_GCC_PAIRS)
    lib = rng.choice(_NATIVE_LIBS)
    commit = rng.choice(_COMMIT_HASHES)
    base_ts = _random_timestamp(rng)

    lines = [
        f"{_ts(base_ts, 0)} [build]   Pipeline started — stage: native_extension_build",
        f"{_ts(base_ts, 1)} [build]   Running: python setup.py build_ext --inplace",
        f"{_ts(base_ts, 2)} [gcc]     gcc -pthread -B /usr/bin/x86_64-linux-gnu-gcc ...",
        f"{_ts(base_ts, 3)} [linker]  /usr/bin/ld: {lib['lib']}: undefined reference to `{lib['symbol']}'",
        f"{_ts(base_ts, 3)} [linker]  /usr/bin/ld: note: '{lib['symbol']}' is defined in DSO /usr/lib/x86_64-linux-gnu/{lib['dso']}",
        f"{_ts(base_ts, 3)} [linker]  ABI mismatch: library compiled with GCC {gcc['old_ver']} ({gcc['old_abi']}), but current compiler is GCC {gcc['new_ver']} ({gcc['new_abi']})",
        f"{_ts(base_ts, 3)} [linker]  This is a known incompatibility — see commit {commit} for the GCC {gcc['new_ver']} ABI patch",
    ]

    if inject_noise:
        lines.append(f"{_ts(base_ts, 4)} [build]   {rng.choice(_NOISE_LINES)}")

    lines.extend([
        f"{_ts(base_ts, 4)} [build]   collect2: error: ld returned 1 exit status",
        f"{_ts(base_ts, 4)} [build]   error: command '/usr/bin/gcc' failed with exit code 1",
        f"{_ts(base_ts, 4)} [build]   EXIT CODE: 1",
        f"{_ts(base_ts, 5)} [build]   Pipeline FAILED at stage: native_extension_build",
        f"{_ts(base_ts, 5)} [cache]   Stale object files detected in build cache — may conflict with new ABI",
        f"{_ts(base_ts, 5)} [env]     _GLIBCXX_USE_CXX11_ABI is not set — compiler flag required for ABI compatibility",
        f"{_ts(base_ts, 6)} [memory]  HINT: ABI mismatch recorded in memory bank. 3-step fix available.",
    ])

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Noise injection helper (mid-episode)
# ---------------------------------------------------------------------------

def generate_noise_line(rng: random.Random) -> str:
    """Return one random CI noise line for mid-episode injection."""
    ts = _random_timestamp(rng)
    return f"{_ts(ts, 0)} [build] {rng.choice(_NOISE_LINES)}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_GENERATORS = {
    "tier_1": _generate_tier1_log,
    "tier_2": _generate_tier2_log,
    "tier_3": _generate_tier3_log,
}


def generate_log(
    tier_id: str,
    rng: random.Random,
    inject_noise: bool = False,
) -> str:
    """
    Generate a procedural CI/CD failure log for the given tier.

    Args:
        tier_id:      One of "tier_1", "tier_2", "tier_3".
        rng:          A seeded random.Random instance.
        inject_noise: Whether to inject irrelevant warning lines.

    Returns:
        A multi-line log string with randomized but semantically-correct content.
    """
    generator = _GENERATORS.get(tier_id)
    if generator is None:
        raise ValueError(f"No procedural generator for tier '{tier_id}'. "
                         f"Valid: {list(_GENERATORS.keys())}")
    return generator(rng, inject_noise=inject_noise)


def generate_memory_hints_tier3(rng: random.Random) -> tuple[list[str], dict]:
    """Generate randomised memory hints and memory bank for Tier 3."""
    gcc = rng.choice(_GCC_PAIRS)
    commit = rng.choice(_COMMIT_HASHES)

    hints = [
        "ABI mismatch recorded in memory bank — 3-step fix available.",
        f"Step 1: Set compiler flag _GLIBCXX_USE_CXX11_ABI=1 (set_env_variable).",
        f"Step 2: Clear stale object files from build cache (clear_cache).",
        f"Step 3: Apply GCC {gcc['new_ver']} ABI patch from commit {commit} (use_memory_fix).",
    ]
    bank = {
        "abi_fix": (
            f"Apply GCC {gcc['new_ver']} ABI patch from commit {commit}: "
            f"1) set _GLIBCXX_USE_CXX11_ABI=1 via set_env_variable, "
            f"2) clear stale build cache via clear_cache, "
            f"3) apply the patch via use_memory_fix."
        ),
    }
    return hints, bank
