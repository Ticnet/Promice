"""
tasks.py — Task definitions for CICDRepairEnv.

Tasks are indexed by tier ID ("tier_1", "tier_2", "tier_3").
Legacy aliases ("easy", "medium", "hard") resolve to the same definitions.

Each task can operate in two modes:
  - **Static** (default): Uses pre-written log strings for deterministic baselines.
  - **Procedural**: Generates fresh, semantically-equivalent logs per episode via
    ``env.procedural.generate_log()``, preventing agent overfitting to exact tokens.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Static CI/CD log templates (preserved for backward-compatible determinism)
# ---------------------------------------------------------------------------

_TIER1_LOG = """\
[2026-04-06T04:10:01Z] [build] Pipeline started — stage: install_packages
[2026-04-06T04:10:02Z] [build] Running: pip install -r requirements.txt
[2026-04-06T04:10:03Z] [build] Collecting numpy==1.24.3
[2026-04-06T04:10:04Z] [build] Collecting pandas==2.0.1
[2026-04-06T04:10:05Z] [build] ERROR: Could not find a version that satisfies the requirement scikit-learn==1.3.0
[2026-04-06T04:10:05Z] [build] ERROR: No matching distribution found for scikit-learn==1.3.0
[2026-04-06T04:10:05Z] [build] ---
[2026-04-06T04:10:06Z] [test]  Running: python -m pytest tests/
[2026-04-06T04:10:07Z] [test]  Traceback (most recent call last):
[2026-04-06T04:10:07Z] [test]    File "tests/test_model.py", line 4, in <module>
[2026-04-06T04:10:07Z] [test]      from sklearn.ensemble import RandomForestClassifier
[2026-04-06T04:10:07Z] [test]  ModuleNotFoundError: No module named 'sklearn'
[2026-04-06T04:10:07Z] [test]  ERROR: failed to collect tests — ModuleNotFoundError
[2026-04-06T04:10:07Z] [build] EXIT CODE: 1
[2026-04-06T04:10:08Z] [build] Pipeline FAILED at stage: install_packages
"""

_TIER2_LOG = """\
[2026-04-06T05:22:11Z] [build]  Pipeline started — stage: dependency_resolution
[2026-04-06T05:22:12Z] [build]  Running: pip install --upgrade pip
[2026-04-06T05:22:13Z] [build]  Running: pip install -r requirements.txt
[2026-04-06T05:22:14Z] [resolver] ERROR: pip's dependency resolver encountered a version conflict.
[2026-04-06T05:22:14Z] [resolver]   Current environment: torch==2.0.0+cpu (cached)
[2026-04-06T05:22:14Z] [resolver]   Requirement: torch==2.1.0
[2026-04-06T05:22:15Z] [cache]   Stale cache detected at ~/.cache/pip/wheels/torch-2.0.0*.whl
[2026-04-06T05:22:15Z] [cache]   Cache timestamp: 2026-03-01 — packages may be outdated
[2026-04-06T05:22:16Z] [resolver] ERROR: Cannot install torch==2.1.0 because of conflicting cached version 2.0.0
[2026-04-06T05:22:16Z] [resolver] HINT: Clear pip cache and pin the correct version in requirements.txt
[2026-04-06T05:22:17Z] [build]  Running: python -c "import torch; print(torch.__version__)"
[2026-04-06T05:22:17Z] [build]  2.0.0+cpu
[2026-04-06T05:22:17Z] [build]  AssertionError: expected torch>=2.1.0, got 2.0.0
[2026-04-06T05:22:18Z] [build]  EXIT CODE: 1
[2026-04-06T05:22:18Z] [build]  Pipeline FAILED at stage: dependency_resolution
"""

_TIER3_LOG = """\
[2026-04-06T06:45:33Z] [build]   Pipeline started — stage: native_extension_build
[2026-04-06T06:45:34Z] [build]   Running: python setup.py build_ext --inplace
[2026-04-06T06:45:35Z] [gcc]     gcc -pthread -B /usr/bin/x86_64-linux-gnu-gcc ...
[2026-04-06T06:45:36Z] [linker]  /usr/bin/ld: libboost_python39.so: undefined reference to `std::__cxx11::basic_string'
[2026-04-06T06:45:36Z] [linker]  /usr/bin/ld: note: 'std::__cxx11::basic_string' is defined in DSO /usr/lib/x86_64-linux-gnu/libstdc++.so.6
[2026-04-06T06:45:36Z] [linker]  ABI mismatch: library compiled with GCC 9 (old ABI), but current compiler is GCC 11 (new ABI)
[2026-04-06T06:45:36Z] [linker]  This is a known incompatibility — see commit abc123 for the GCC 11 ABI patch
[2026-04-06T06:45:37Z] [build]   collect2: error: ld returned 1 exit status
[2026-04-06T06:45:37Z] [build]   error: command '/usr/bin/gcc' failed with exit code 1
[2026-04-06T06:45:37Z] [build]   EXIT CODE: 1
[2026-04-06T06:45:38Z] [build]   Pipeline FAILED at stage: native_extension_build
[2026-04-06T06:45:38Z] [cache]   Stale object files detected in build cache — may conflict with new ABI
[2026-04-06T06:45:38Z] [env]     _GLIBCXX_USE_CXX11_ABI is not set — compiler flag required for ABI compatibility
[2026-04-06T06:45:39Z] [memory]  HINT: ABI mismatch recorded in memory bank. 3-step fix available.
"""


# ---------------------------------------------------------------------------
# Task registry — canonical keys: tier_1, tier_2, tier_3
# ---------------------------------------------------------------------------

TASKS: dict[str, dict] = {
    "tier_1": {
        "description": (
            "A Python dependency is missing from the environment. "
            "The pipeline fails during the install_packages stage with a ModuleNotFoundError. "
            "Fix: install the missing dependency."
        ),
        "pipeline_stage":    "install_packages",
        "failure_type":      "missing_dependency",
        "error_type":        "ModuleNotFoundError",
        "failure_log":       _TIER1_LOG,
        "required_sequence": [1],          # install_dependency
        "optimal_steps":     1,
        "max_steps":         5,
        "memory_bank":       {},
        "memory_hints":      [],
        "procedural":        True,
    },
    "tier_2": {
        "description": (
            "A stale pip cache is causing a version conflict for a required package. "
            "The pipeline fails during dependency_resolution. "
            "Fix: clear the cache first, then change/pin the correct version."
        ),
        "pipeline_stage":    "dependency_resolution",
        "failure_type":      "cache_version_conflict",
        "error_type":        "VersionConflictError",
        "failure_log":       _TIER2_LOG,
        "required_sequence": [4, 2],       # clear_cache → change_version
        "optimal_steps":     2,
        "max_steps":         8,
        "memory_bank":       {},
        "memory_hints": [
            "Cache conflict detected — stale pip wheel for a required package.",
            "Pinned version mismatch: requirements expect a newer version.",
        ],
        "procedural":        True,
    },
    "tier_3": {
        "description": (
            "A C++ ABI mismatch between GCC versions prevents the native extension from linking. "
            "The fix requires a precise 3-step sequence: first set the compiler ABI flag, "
            "then clear stale build artifacts, and finally apply the ABI patch from the memory bank. "
            "Agents must read logs, interpret memory hints, and execute actions in strict order."
        ),
        "pipeline_stage":    "native_extension_build",
        "failure_type":      "abi_mismatch",
        "error_type":        "ABILinkError",
        "failure_log":       _TIER3_LOG,
        "required_sequence": [3, 4, 6],    # set_env_variable → clear_cache → use_memory_fix
        "optimal_steps":     3,
        "max_steps":         10,
        "memory_bank": {
            "abi_fix": (
                "Apply GCC 11 ABI patch from commit abc123: "
                "1) set _GLIBCXX_USE_CXX11_ABI=1 via set_env_variable, "
                "2) clear stale build cache via clear_cache, "
                "3) apply the patch via use_memory_fix."
            ),
        },
        "memory_hints": [
            "ABI mismatch recorded in memory bank — previous fix available.",
            "Step 1: Set compiler flag _GLIBCXX_USE_CXX11_ABI=1 (set_env_variable).",
            "Step 2: Clear stale object files from build cache (clear_cache).",
            "Step 3: Apply GCC 11 ABI patch from commit abc123 (use_memory_fix).",
        ],
        "procedural":        True,
    },
}

# ---------------------------------------------------------------------------
# Backward-compatible aliases: easy / medium / hard → tier_1 / tier_2 / tier_3
# ---------------------------------------------------------------------------

_ALIASES: dict[str, str] = {
    "easy":   "tier_1",
    "medium": "tier_2",
    "hard":   "tier_3",
}

# Populate aliases so TASKS["easy"] is TASKS["tier_1"] etc.
for _alias, _canonical in _ALIASES.items():
    TASKS[_alias] = TASKS[_canonical]

# All valid tier IDs (canonical only, no aliases)
TIER_IDS: tuple[str, ...] = ("tier_1", "tier_2", "tier_3")

# Canonical + alias lookup
ALL_TASK_IDS: tuple[str, ...] = tuple(TASKS.keys())


def resolve_task_id(task_id: str) -> str:
    """Resolve an alias to its canonical tier ID, or return as-is."""
    return _ALIASES.get(task_id, task_id)
