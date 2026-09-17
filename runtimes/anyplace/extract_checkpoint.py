"""Extract only the multitask model from the official release archive."""

import sys
import zipfile
from pathlib import Path


def extract(archive: Path, destination: Path):
    member = "anyplace_ckpts/anyplace_multitask/model.pth"
    with zipfile.ZipFile(archive) as bundle:
        content = bundle.read(member)  # fixed path; never extract arbitrary archive paths
    checkpoint = destination / "anyplace_multitask/model.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists() or checkpoint.read_bytes() != content:
        temporary = checkpoint.with_suffix(".pth.partial")
        temporary.write_bytes(content)
        temporary.replace(checkpoint)
    print(f"Checkpoint: {checkpoint} ({len(content)} bytes)")


if __name__ == "__main__":
    extract(Path(sys.argv[1]), Path(sys.argv[2]))
