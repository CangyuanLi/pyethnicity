import subprocess
import time


def main():
    with open("src/pyethnicity/__init__.py") as f:
        version = f.readlines()[0].split("=")[-1].strip().strip('"')

    print("committing version changes")
    subprocess.run(["git", "commit", "-am", version], check=True)

    print("pushing to remote")
    subprocess.run(["git", "push", "origin"], check=True)
    time.sleep(2)

    print("creating release")
    subprocess.run(
        ["gh", "release", "create", f"v{version}", "--generate-notes"], check=True
    )

    print("uploading to pypi")
    subprocess.run(["pyproject", "upload"])


if __name__ == "__main__":
    main()
