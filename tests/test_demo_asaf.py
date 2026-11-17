"""La démo doit tourner sans données ni modèle entraîné (vérification d'installation)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_demo_selftest() -> None:
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "demo_asaf.py"), "--selftest", "--device", "cpu"],
        capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, r.stderr
    assert "selftest OK" in r.stdout


def test_published_checkpoint_loads() -> None:
    ckpt = ROOT / "checkpoints" / "asaf_net_v2_vigs.pth"
    if not ckpt.exists():
        import pytest

        pytest.skip("poids publiés non téléchargés")
    sys.path.insert(0, str(ROOT))
    from depth_fusion.models.asaf_net import ASAFNetV2, ASAFNetV2Config
    from depth_fusion.utils.checkpoint import load_checkpoint

    load_checkpoint(ckpt, ASAFNetV2(ASAFNetV2Config(pretrained=False)))
