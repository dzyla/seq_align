"""Local gate: byte-compile all first-party code and run the test suite. Exits non-zero on failure."""
import pathlib
import py_compile
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent


def main() -> int:
    sources = [ROOT / "app.py", *sorted((ROOT / "modules").glob("*.py"))]
    for path in sources:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            print(f"Compile error: {exc}")
            return 1
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)
    if result.returncode != 0:
        print("Pre-commit failed.")
        return result.returncode
    print("Pre-commit passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
