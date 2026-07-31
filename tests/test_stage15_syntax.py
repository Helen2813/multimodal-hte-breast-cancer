from pathlib import Path
import py_compile


def test_stage15_scripts_compile():
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "scripts").glob("*.py")):
        py_compile.compile(str(path), doraise=True)
