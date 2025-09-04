import json
import numpy as np
from click.testing import CliRunner
import pytest

from dpf2.cli.main import main
from dpf2.ai import simple_surrogates


def test_make_surrogate_cli(tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    data = np.column_stack((np.linspace(90, 200, 5), np.linspace(1, 2, 5)))
    csv_path = tmp_path / "data.csv"
    np.savetxt(csv_path, data, delimiter=",")
    outdir = tmp_path / "models"
    runner = CliRunner()
    result = runner.invoke(main, ["make-surrogate", "--data", str(csv_path), "--outdir", str(outdir)])
    assert result.exit_code == 0
    assert (outdir / "yield_model.json").exists()
    assert (outdir / "yield_model.onnx").exists()
    monkeypatch.setattr(simple_surrogates, "MODEL_DIR", outdir)
    model = simple_surrogates.load_yield_surrogate()
    val = float(data[0, 0])
    pred = model.predict(val)
    assert model.domain[0] <= val <= model.domain[1]
    with pytest.raises(simple_surrogates.OutOfDomainError):
        model.predict(model.domain[1] + 10)
