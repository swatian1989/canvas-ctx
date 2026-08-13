"""Shared config loading: resolves the `inherit:` chain used by config/*.yaml.

config/crc_train_brca_apply.yaml inherits config/pilot.yaml, which inherits
config/default.yaml; each level overrides only the keys it changes. The
existing stage1/stage6 runners predate this chain and load their single file
with a plain `yaml.safe_load`, so they only see keys defined directly in
whatever file they were pointed at. Any new runner that needs the full merged
config (anything reading, e.g., `patching`/`model` from
crc_train_brca_apply.yaml, which only overrides `cn_discovery`/`paired`/
`inference`/`clinical`) should use `load_config` instead.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if key == "inherit":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict:
    """Load a config YAML, recursively resolving `inherit:`.

    `inherit:` values are repo-root-relative paths (e.g. `config/pilot.yaml`),
    matching how they're written in the config files themselves.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    parent_rel = raw.get("inherit")
    if not parent_rel:
        return raw
    parent = load_config(_REPO_ROOT / parent_rel)
    return _deep_merge(parent, raw)
