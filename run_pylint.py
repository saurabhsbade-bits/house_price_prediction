import sys
import subprocess

def main():
    cmd = [sys.executable, "-m", "pylint", "--output-format=json", "app/", "api/", "tests/"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout or ""
    err = proc.stderr or ""
    with open("pylint_report.json", "w", encoding="utf-8") as f:
        f.write(out)
    # Also write a human-readable fallback
    with open("pylint_report_fallback.txt", "w", encoding="utf-8") as f:
        f.write("STDOUT:\n")
        f.write(out)
        f.write("\nSTDERR:\n")
        f.write(err)
    print(f"Wrote pylint_report.json and pylint_report_fallback.txt (rc={proc.returncode})")

if __name__ == "__main__":
    main()
