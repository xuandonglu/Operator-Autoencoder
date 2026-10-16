"""Create a source-only release archive without changing the working tree."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = ("README.md", "requirements.txt", "CITATION.cff", ".gitignore")


def source_files():
    yield from (ROOT / name for name in ROOT_FILES)
    for folder in ("reduction", "latent_pred", "tools"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".md"}:
                yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "Operator-Autoencoder-source.zip")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = list(source_files())
    with ZipFile(args.output, "w", compression=ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(f"Exported {len(paths)} source files to {args.output.resolve()}")


if __name__ == "__main__":
    main()
