"""What the numerical core loads, and whether each runtime dependency is used by the package.

CXD's core computes on arrays: it should run without a 3-D viewer, a plotting library, a SPEC
parser or QoraPhase. Known gaps are ``xfail(strict=True)``, so a fix turns the test red and the
mark has to go in the same change.
"""

import ast
import subprocess
import sys
import tomllib
from functools import cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "cxd"
NUMERICAL = [
    "cxd.core.handoff",
    "cxd.core.rocking",
    "cxd.core.statistics",
    "cxd.core.transform",
    "cxd.mapping.analysis",
    "cxd.mapping.profiles",
    "cxd.mapping.qspace",
    "cxd.mapping.spherical",
    "cxd.twin_finder",
]
PRESENTATION = {"matplotlib", "pyvista", "vtk"}
# CXD modules that import qorageom, whose package init loads qorageom.spec through scan_geometry
VIA_QORAGEOM = {"cxd.core.rocking", "cxd.core.transform", "cxd.mapping.qspace"}
IMPORT_NAMES = {"roi-rectangle": "roi_rectangle", "scikit-learn": "sklearn"}


@cache
def loaded_after(module: str) -> set[str]:
    code = f"import sys, {module}\nprint('\\n'.join(sys.modules))"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    return {name.split(".")[0] for name in out.split()} | set(out.split())


def strict_xfail(reason):
    return pytest.mark.xfail(strict=True, reason=reason)


@pytest.mark.parametrize(
    "module",
    NUMERICAL,
)
def test_numerical_module_loads_no_presentation_library(module):
    assert not loaded_after(module) & PRESENTATION


@pytest.mark.parametrize(
    "module",
    [
        pytest.param(m, marks=strict_xfail("import qorageom loads qorageom.spec"))
        if m in VIA_QORAGEOM
        else m
        for m in NUMERICAL
    ],
)
def test_numerical_module_loads_no_spec_parser(module):
    assert "qorageom.spec" not in loaded_after(module)


def test_package_never_loads_qoraphase():
    modules = sorted(
        "cxd." + ".".join(p.relative_to(PACKAGE).with_suffix("").parts)
        for p in PACKAGE.rglob("*.py")
        if p.name != "__init__.py"
    )
    assert "qoraphase" not in loaded_after(", ".join(modules))


def package_imports() -> set[str]:
    names = set()
    for path in PACKAGE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names.add(node.module.split(".")[0])
    return names


def runtime_dependencies() -> list[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    return [
        dep.split(" @ ")[0].split("[")[0].split(">")[0].split("=")[0].strip()
        for dep in project["dependencies"]
    ]


SCRIPT_ONLY = strict_xfail("imported only by scripts/ or examples/, not by the package")


@pytest.mark.parametrize(
    "dist",
    [
        pytest.param(d, marks=SCRIPT_ONLY) if d in {"roi-rectangle", "tifffile"} else d
        for d in runtime_dependencies()
    ],
)
def test_runtime_dependency_is_imported_by_the_package(dist):
    assert IMPORT_NAMES.get(dist, dist) in package_imports()
