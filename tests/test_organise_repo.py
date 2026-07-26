"""organise_repo idempotence on a temp copy of a mock old-layout repo."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

OLD_FILES = [
    "network_conversion/cim_to_network_json.py",
    "network_conversion/batch_convert.py",
    "network_conversion/extract_lv_network.py",
    "network_conversion/README.md",
    "network_conversion/GOLDCR_8HB_LEXCEN_network.json",
    "examples/run_scenario.py",
    "examples/run_doe_feeder.py",
    "examples/RUN_DOE_GUIDE.md",
    "examples/disaggregate_transformer.py",
    "examples/scenario_2/data_translation/wide_to_long_translator.py",
    "examples/scenario_2/data_translation/lexcen_data.csv",
    "examples/scenario_2/break",
    "examples/scenario_doe_output/doe.csv",
    "results/doe.csv",
    "bin/micromamba",
    "output.log",
    "src/converge_soe/doe_solver.py",
    "src/converge_soe/__pycache__/x.pyc",
]


def make_mock(tmp_path):
    for f in OLD_FILES:
        p = tmp_path / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("placeholder\n")
    (tmp_path / "examples/scenario_doe").mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "examples/scenario_doe/transformer_params.json",
                tmp_path / "examples/scenario_doe/transformer_params.json")
    (tmp_path / "scripts").mkdir(exist_ok=True)
    shutil.copy(REPO / "scripts/organise_repo.py",
                tmp_path / "scripts/organise_repo.py")
    return tmp_path


def run(tmp_path, *args):
    return subprocess.run([sys.executable, "scripts/organise_repo.py", *args],
                          cwd=tmp_path, capture_output=True, text=True,
                          timeout=120)


def test_dry_run_changes_nothing(tmp_path):
    make_mock(tmp_path)
    before = sorted(str(p.relative_to(tmp_path))
                    for p in tmp_path.rglob("*") if p.is_file())
    r = run(tmp_path)
    assert r.returncode == 0
    assert "DRY RUN" in r.stdout
    after = sorted(str(p.relative_to(tmp_path))
                   for p in tmp_path.rglob("*") if p.is_file())
    assert before == after


def test_apply_then_noop(tmp_path):
    make_mock(tmp_path)
    r1 = run(tmp_path, "--apply")
    assert r1.returncode == 0, r1.stderr
    # moved to the right places
    assert (tmp_path / "src/converge_soe/network/cim_to_json.py").exists()
    assert (tmp_path / "examples/legacy/run_doe_feeder.py").exists()
    assert (tmp_path / "tools/disaggregate_transformer.py").exists()
    assert (tmp_path / "data/meter/lexcen_data.csv").exists()
    assert (tmp_path / "archive/reference_outputs/scenario_doe_output/doe.csv").exists()
    assert (tmp_path / "archive/bin/micromamba").exists()
    assert (tmp_path / "config/transformers/distribution_onan.yaml").exists()
    # deleted
    assert not (tmp_path / "output.log").exists()
    assert not (tmp_path / "examples/scenario_2/break").exists()
    assert not (tmp_path / "src/converge_soe/__pycache__").exists()
    # kept for the legacy example
    assert (tmp_path / "examples/scenario_doe/transformer_params.json").exists()
    # idempotent
    r2 = run(tmp_path)
    assert "Nothing to do" in r2.stdout
