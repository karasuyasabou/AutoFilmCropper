"""Build release archives for the desktop app with PyInstaller."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import PyInstaller.__main__

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_support.app_metadata import (  # noqa: E402
    APP_DESCRIPTION,
    APP_NAME,
    BUNDLE_ID,
    DEFAULT_VERSION,
)


PLATFORM_CHOICES = {"macos-arm64", "macos-intel", "windows-x64"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=DEFAULT_VERSION, help="Release version or tag.")
    parser.add_argument(
        "--platform",
        required=True,
        choices=sorted(PLATFORM_CHOICES),
        help="Release platform label used in the archive name.",
    )
    return parser.parse_args()


def normalized_version(version: str) -> str:
    return version[1:] if version.startswith("v") else version


def generate_icons() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_icons.py")],
        check=True,
        cwd=ROOT,
    )


def icon_path_for_platform(platform_name: str) -> Path:
    icons_dir = ROOT / "build" / "icons"
    if platform_name.startswith("macos"):
        return icons_dir / "app-icon.icns"
    if platform_name.startswith("windows"):
        return icons_dir / "app-icon.ico"
    return icons_dir / "app-icon-1024.png"


def build_with_pyinstaller(platform_name: str) -> Path:
    build_root = ROOT / "build" / "pyinstaller" / platform_name
    dist_root = ROOT / "dist" / platform_name
    spec_root = ROOT / "build" / "spec" / platform_name

    for path in (build_root, dist_root, spec_root):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    icon_path = icon_path_for_platform(platform_name)
    args = [
        str(ROOT / "main.py"),
        "--noconfirm",
        "--clean",
        "--windowed",
        "--name",
        APP_NAME,
        "--distpath",
        str(dist_root),
        "--workpath",
        str(build_root),
        "--specpath",
        str(spec_root),
        "--icon",
        str(icon_path),
        "--hidden-import",
        "tifffile",
    ]
    if sys.platform == "darwin":
        args.extend(["--osx-bundle-identifier", BUNDLE_ID])

    PyInstaller.__main__.run(args)

    bundle_name = f"{APP_NAME}.app" if sys.platform == "darwin" else APP_NAME
    return dist_root / bundle_name


def archive_bundle(bundle_path: Path, version: str, platform_name: str) -> Path:
    release_dir = ROOT / "release-assets"
    release_dir.mkdir(parents=True, exist_ok=True)

    archive_base = release_dir / f"{APP_NAME}-{version}-{platform_name}"
    if archive_base.with_suffix(".zip").exists():
        archive_base.with_suffix(".zip").unlink()

    archive_file = shutil.make_archive(
        str(archive_base),
        "zip",
        root_dir=bundle_path.parent,
        base_dir=bundle_path.name,
    )
    return Path(archive_file)


def main() -> int:
    args = parse_args()
    version = normalized_version(args.version)

    print(f"Building {APP_NAME} {version} for {args.platform}")
    print(APP_DESCRIPTION)

    generate_icons()
    bundle_path = build_with_pyinstaller(args.platform)
    archive_path = archive_bundle(bundle_path, version, args.platform)

    print(f"bundle: {bundle_path}")
    print(f"archive: {archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
