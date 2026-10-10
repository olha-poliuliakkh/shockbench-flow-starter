"""The scoring container's image as package data: its Dockerfile, its hash-pinned requirements and its build context.

``IMAGE_DIR`` holds ``Dockerfile`` and ``requirements.txt``, the files the organisers build the policy image from.
``build_context()`` is the tar the image builds from: those two files, ``launch.py`` (the container's entry point,
``shockbench_flow/hosting/launch.py``) and ``lib/`` holding the kit's two packages, ``shockbench_flow`` and
``shockbench_flow_agent``, as installed here (the wheel's files, bytecode left out). ``build_image(tag)`` builds it
with Docker and returns the image's ID; ``image_distributions()`` lists the distributions it installs, by name and
version, and ``image_requirements()`` the file's text.

An image built here runs the kit of the installed wheel, the organisers' image the kit of the version they built it
from; the scoring rules the two apply are the same (``shockbench_flow_agent.LIMITS``). The torch pin is the
linux/amd64 wheel, so the image is linux/amd64 (emulated on an ARM machine, and slower there).
"""

import io
import re
import subprocess
import tarfile
from pathlib import Path

import shockbench_flow
import shockbench_flow_agent


IMAGE_DIR = Path(__file__).resolve().parent / "policy_image"
DOCKERFILE, REQUIREMENTS = IMAGE_DIR / "Dockerfile", IMAGE_DIR / "requirements.txt"
LAUNCHER = Path(shockbench_flow.__file__).resolve().parent / "hosting" / "launch.py"
KIT_PACKAGES = ("shockbench_flow", "shockbench_flow_agent")  # what the image's lib/ holds
SHIM_MODULE = "shockbench_flow_agent"  # the module the launcher runs (the Dockerfile's SHIM_MODULE)
PLATFORM = "linux/amd64"
DEFAULT_TAG = "sbf-policy:local"
_REQUIREMENT = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:==\s*([^\s\\;]+)|@\s*\S+)")
_TORCH_VERSION = re.compile(r"torch-([0-9][^-]*)-")


def image_requirements() -> str:
    """The text of the image's ``requirements.txt`` (every wheel pinned by its hash)."""
    return REQUIREMENTS.read_text()


def image_distributions() -> dict[str, str]:
    """The distributions the image installs, name -> version (torch's from its wheel's URL), in the file's order."""
    out = {}
    for line in image_requirements().splitlines():
        m = _REQUIREMENT.match(line)
        if m:
            version = m[2]
            if version is None:  # a direct URL: the version is in the wheel's file name
                found = _TORCH_VERSION.search(line.replace("%2B", "+"))
                version = found[1] if found else None
            out[m[1]] = version
    return out


def _package_dirs() -> dict[str, Path]:
    return {
        "shockbench_flow": Path(shockbench_flow.__file__).resolve().parent,
        "shockbench_flow_agent": Path(shockbench_flow_agent.__file__).resolve().parent,
    }


def _add(tar: tarfile.TarFile, arcname: str, data: bytes) -> None:
    info = tarfile.TarInfo(arcname)
    info.size, info.mode, info.mtime = len(data), 0o644, 0
    tar.addfile(info, io.BytesIO(data))


def build_context() -> bytes:
    """The image's build context as tar bytes (module docstring): the same files give the same bytes.

    Members: ``Dockerfile``, ``requirements.txt``, ``launch.py`` and ``lib/<package>/...`` for ``KIT_PACKAGES``, every
    file of the installed packages but ``__pycache__`` and compiled bytecode, sorted, with fixed times and modes.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tar:
        _add(tar, "Dockerfile", DOCKERFILE.read_bytes())
        _add(tar, "requirements.txt", REQUIREMENTS.read_bytes())
        _add(tar, "launch.py", LAUNCHER.read_bytes())
        for name, root in _package_dirs().items():
            for path in sorted(root.rglob("*")):
                rel = path.relative_to(root)
                if path.is_file() and "__pycache__" not in rel.parts and path.suffix not in (".pyc", ".pyo"):
                    _add(tar, f"lib/{name}/{rel.as_posix()}", path.read_bytes())
    return buf.getvalue()


def build_image(tag: str = DEFAULT_TAG, *, docker: tuple[str, ...] = ("docker",), quiet: bool = True) -> str:
    """Build the image from ``build_context()`` with ``docker build`` and return its ID (``sha256:<hex>``).

    The first build downloads the base image and the wheels (torch and SciPy: some hundreds of MB); later builds come
    from Docker's cache. ``quiet`` False shows Docker's build output.

    Raises:
        FileNotFoundError: when the ``docker`` command is not found.
        subprocess.CalledProcessError: when the build or the inspection of the image fails (Docker's daemon not
            running, no network on a first build).

    """
    argv = [*docker, "build", "--platform", PLATFORM, "--build-arg", f"SHIM_MODULE={SHIM_MODULE}", "-t", tag, "-"]
    if quiet:
        argv.insert(len(docker) + 1, "-q")
    subprocess.run(argv, input=build_context(), check=True, stdout=subprocess.DEVNULL if quiet else None)
    out = subprocess.run(
        [*docker, "image", "inspect", "--format", "{{.Id}}", tag], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


__all__ = [
    "DOCKERFILE",
    "IMAGE_DIR",
    "REQUIREMENTS",
    "build_context",
    "build_image",
    "image_distributions",
    "image_requirements",
]
