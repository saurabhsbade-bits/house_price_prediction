import sys
import subprocess

def main():
    cmd = [sys.executable, "-m", "pytest", "-v"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout or ""
    err = proc.stderr or ""
    with open("test_run_output.txt", "w", encoding="utf-8") as f:
        f.write("STDOUT:\n")
        f.write(out)
        f.write("\nSTDERR:\n")
        f.write(err)
    print(f"Wrote test output to test_run_output.txt (rc={proc.returncode})")

if __name__ == "__main__":
    main()
