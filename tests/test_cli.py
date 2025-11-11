import subprocess
import sys
from pathlib import Path


def test_bbox_parser_help():
    # Ensure CLI shows help when no args; basic smoke test
    p = Path(__file__).resolve().parents[1] / "src" / "map_postcodes.py"
    proc = subprocess.run([sys.executable, str(p), "--help"], capture_output=True)
    assert proc.returncode == 0
    assert b"Create a postcode line map" in proc.stdout or b"usage:" in proc.stdout
