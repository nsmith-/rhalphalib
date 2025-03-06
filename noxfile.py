from __future__ import annotations

import argparse
from pathlib import Path
import sys

import nox

DIR = Path(__file__).parent.resolve()

nox.needs_version = ">=2024.3.2"
nox.options.sessions = ["lint", "tests"]
nox.options.default_venv_backend = "uv|virtualenv"


@nox.session(python=["3.8", "3.10"])
def lint(session: nox.Session) -> None:
    """
    Run the linter.
    """
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", "--show-diff-on-failure", *session.posargs)


@nox.session(reuse_venv=True, venv_backend="conda")
@nox.parametrize(
    "python,root",
    [
        ("3.9", "6.22.08"),
        ("3.10", "6.30.04"),
        ("3.10", "6.32.10"),
    ],
)
def tests(session: nox.Session, root: str) -> None:
    """
    Run the unit and regular tests.
    """
    if root == "6.22.08" and sys.platform == "darwin":
        session.skip("ROOT 6.22.08 is not working on macOS")
    session.conda_install(f"root=={root}")
    session.install(".[test]")
    session.run("pytest", *session.posargs)


@nox.session(reuse_venv=True)
def docs(session: nox.Session) -> None:
    """
    Build the docs. Pass --non-interactive to avoid serving. First positional argument is the target directory.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", dest="builder", default="html", help="Build target (default: html)")
    parser.add_argument("output", nargs="?", help="Output directory")
    args, posargs = parser.parse_known_args(session.posargs)
    serve = args.builder == "html" and session.interactive

    session.install("-e.[docs]", "sphinx-autobuild")

    shared_args = (
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        f"-b={args.builder}",
        "docs",
        args.output or f"docs/_build/{args.builder}",
        *posargs,
    )

    if serve:
        session.run("sphinx-autobuild", "--open-browser", *shared_args)
    else:
        session.run("sphinx-build", "--keep-going", *shared_args)


@nox.session
def build_api_docs(session: nox.Session) -> None:
    """
    Build (regenerate) API docs.
    """

    session.install("sphinx")
    session.run(
        "sphinx-apidoc",
        "-o",
        "docs/api/",
        "--module-first",
        "--no-toc",
        "--force",
        "src/rhalphalib",
    )
