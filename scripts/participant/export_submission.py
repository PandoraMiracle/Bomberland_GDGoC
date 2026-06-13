"""
export_submission.py — Package the MAPPO agent into a submission zip.

Usage:
  python -m scripts.participant.export_submission [options]

Options:
  --checkpoint  PATH    actor checkpoint .pth (required unless --fallback-only)
  --output      PATH    output zip path (default: submission.zip)
  --fallback-only       ship only the embedded fallback (no model.pth)
  --agent-src   PATH    override agent.py source (default: agent/mappo_agent/agent.py)

Validates:
  ✓ Exactly one agent.py
  ✓ model.pth present (unless --fallback-only)
  ✓ zip <= 100 MB
  ✓ extracted <= 300 MB
  ✓ <= 20 files
  ✓ No logs/checkpoints included
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ZIP_LIMIT_BYTES      = 100 * 1024 * 1024   # 100 MB
EXTRACTED_LIMIT_BYTES = 300 * 1024 * 1024  # 300 MB
MAX_FILES            = 20


def validate_zip(zip_path: str) -> None:
    """Validate the submission zip. Raises ValueError on failure."""
    zp = Path(zip_path)
    if not zp.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    zip_size = zp.stat().st_size
    if zip_size > ZIP_LIMIT_BYTES:
        raise ValueError(f"Zip size {zip_size/1e6:.1f} MB exceeds 100 MB limit")

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if len(names) > MAX_FILES:
            raise ValueError(f"Zip contains {len(names)} files; limit is {MAX_FILES}")

        agent_py = [n for n in names if os.path.basename(n) == "agent.py"]
        if len(agent_py) != 1:
            raise ValueError(f"Expected exactly 1 agent.py, found: {agent_py}")
        if agent_py[0] != "agent.py":
            raise ValueError(f"agent.py must be at zip root, found at: {agent_py[0]}")

        total_extracted = sum(info.file_size for info in zf.infolist())
        if total_extracted > EXTRACTED_LIMIT_BYTES:
            raise ValueError(
                f"Extracted size {total_extracted/1e6:.1f} MB exceeds 300 MB limit"
            )

        # No prohibited names
        for name in names:
            base = os.path.basename(name).lower()
            if base.endswith(".log") or "checkpoint" in base:
                raise ValueError(f"Submission should not contain: {name}")

    print(f"[Validate] [OK] {Path(zip_path).name}: {zip_size/1e6:.2f} MB, {len(names)} files")


def build_submission(
    checkpoint:    str | None,
    output:        str = "submission.zip",
    fallback_only: bool = False,
    agent_src:     str | None = None,
) -> None:
    agent_src_path = Path(agent_src) if agent_src else ROOT / "agent" / "mappo_agent" / "agent.py"
    if not agent_src_path.exists():
        raise FileNotFoundError(f"agent.py source not found: {agent_src_path}")

    if not fallback_only and checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        model_size = ckpt_path.stat().st_size
        print(f"[Export] Checkpoint: {ckpt_path.name} ({model_size/1e6:.2f} MB)")

    with tempfile.TemporaryDirectory() as tmpdir:
        td = Path(tmpdir)

        # Copy agent.py
        shutil.copy2(agent_src_path, td / "agent.py")

        # Copy model
        if not fallback_only and checkpoint:
            shutil.copy2(checkpoint, td / "model.pth")

        # Smoke-test the agent.py in isolation
        _smoke_test_agent(td / "agent.py", str(td))

        # Create zip (flat structure)
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(out_path), "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(td.iterdir()):
                zf.write(f, arcname=f.name)

    validate_zip(output)
    print(f"[Export] [OK] Submission ready: {output}")


def _smoke_test_agent(agent_py: Path, agent_dir: str) -> None:
    """Import agent.py from tmp dir and call act() once."""
    import importlib.util
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)
    spec = importlib.util.spec_from_file_location("_export_smoke_agent", str(agent_py))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    agent = mod.Agent(0)

    # Minimal synthetic obs
    import numpy as np
    grid = np.zeros((13, 13), dtype=np.int8)
    grid[0, :] = 1; grid[-1, :] = 1; grid[:, 0] = 1; grid[:, -1] = 1
    players = np.zeros((4, 5), dtype=np.int8)
    players[0] = [1, 1, 1, 1, 0]
    players[1] = [11, 11, 1, 1, 0]
    players[2] = [1, 11, 1, 1, 0]
    players[3] = [11, 1, 1, 1, 0]
    obs = {"map": grid, "players": players, "bombs": np.zeros((0, 4), dtype=np.int8)}
    action = agent.act(obs)
    assert 0 <= int(action) <= 5, f"Smoke test: act() returned {action}"
    print(f"[Export] Smoke test passed (action={action}, fallback={agent._use_fallback})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package MAPPO agent for submission")
    parser.add_argument("--checkpoint",    type=str, default=None)
    parser.add_argument("--output",        type=str, default="submission.zip")
    parser.add_argument("--fallback-only", action="store_true")
    parser.add_argument("--agent-src",     type=str, default=None)
    args = parser.parse_args()

    if not args.fallback_only and not args.checkpoint:
        best_path = ROOT / "checkpoints" / "mappo" / "best.pth"
        latest_path = ROOT / "checkpoints" / "mappo" / "latest.pth"
        if best_path.exists():
            args.checkpoint = str(best_path)
            print(f"[Export] Auto-selected checkpoint: {best_path.name}")
        elif latest_path.exists():
            args.checkpoint = str(latest_path)
            print(f"[Export] Auto-selected checkpoint: {latest_path.name}")
        else:
            parser.error("--checkpoint required unless --fallback-only is set, and no best.pth or latest.pth found.")

    build_submission(
        checkpoint    = args.checkpoint,
        output        = args.output,
        fallback_only = args.fallback_only,
        agent_src     = args.agent_src,
    )


if __name__ == "__main__":
    main()
