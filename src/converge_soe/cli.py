"""Console entry points (csoe-build / csoe-run / csoe-analyse)."""
import runpy
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"


def _run(name):
    sys.argv[0] = str(_SCRIPTS / name)
    runpy.run_path(str(_SCRIPTS / name), run_name="__main__")


def build_main():
    _run("build_network.py")


def run_main():
    _run("run_feeder.py")


def analyse_main():
    _run("analyse_results.py")
