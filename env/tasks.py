"""
tasks.py — Task definitions for CICDRepairEnv.

Tasks are indexed by tier ID ("tier_1", "tier_2", "tier_3").
Legacy aliases ("easy", "medium", "hard") resolve to the same definitions.

All logs are now procedurally generated via ``env.procedural.generate_log()``,
preventing agent overfitting to exact tokens.
"""

from __future__ import annotations

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
        "required_sequence": [1],          # install_dependency
        "optimal_steps":     1,
        "max_steps":         5,
        "memory_bank":       {},
        "memory_hints":      [],
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
        "required_sequence": [4, 2],       # clear_cache → change_version
        "optimal_steps":     2,
        "max_steps":         8,
        "memory_bank":       {},
        "memory_hints": [
            "Cache conflict detected — stale pip wheel for a required package.",
            "Pinned version mismatch: requirements expect a newer version.",
        ],
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
