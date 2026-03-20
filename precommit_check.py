import subprocess
try:
    subprocess.run(["python", "-m", "py_compile", "app.py", "modules/clustering.py", "modules/pdb_renumber.py", "tests/test_pdb_renumber.py"], check=True)
    subprocess.run(["python", "-m", "unittest", "discover", "tests"], check=True)
    print("Pre-commit passed.")
except subprocess.CalledProcessError as e:
    print(f"Pre-commit failed: {e}")
