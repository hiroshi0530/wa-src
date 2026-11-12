"""Execute the data-free examples supported by the shared uv environment."""

import json
import sys
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "environment": "main_314.ipynb",
    "sympy": "article/library/sympy/01/01_nb.ipynb",
    "statistics": "rec/stats/01/01_nb.ipynb",
    "linear-algebra": "rec/linalg/base/base_nb.ipynb",
}


@pytest.mark.parametrize("notebook_path", NOTEBOOKS.values(), ids=NOTEBOOKS.keys())
def test_notebook(notebook_path, tmp_path, monkeypatch):
    """Run a copy with the current interpreter and retain outputs on failure."""
    for variable, directory in {
        "IPYTHONDIR": "ipython",
        "JUPYTER_CONFIG_DIR": "jupyter-config",
        "JUPYTER_RUNTIME_DIR": "jupyter-runtime",
        "MPLCONFIGDIR": "matplotlib",
    }.items():
        config_dir = tmp_path / directory
        config_dir.mkdir(mode=0o700)
        monkeypatch.setenv(variable, str(config_dir))
    monkeypatch.setenv("MPLBACKEND", "Agg")

    # A private kernelspec avoids relying on a user-registered wa-src kernel.
    jupyter_dir = tmp_path / "jupyter"
    kernel_dir = jupyter_dir / "kernels" / "wa-src-ci"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    "{connection_file}",
                ],
                "display_name": "wayama.io CI",
                "language": "python",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("JUPYTER_PATH", str(jupyter_dir))

    notebook = nbformat.read(ROOT / notebook_path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        cell.outputs = []
        cell.execution_count = None
        # This standalone cell reports macOS metadata, not a calculation.
        if sys.platform != "darwin" and cell.source.strip() == "!sw_vers":
            cell.metadata.setdefault("tags", []).append("skip-execution")

    executed_path = tmp_path / "executed" / notebook_path
    executed_path.parent.mkdir(parents=True)
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="wa-src-ci",
        force_raise_errors=True,
        resources={"metadata": {"path": str(executed_path.parent)}},
    )
    try:
        client.execute()
    finally:
        nbformat.write(notebook, executed_path)
