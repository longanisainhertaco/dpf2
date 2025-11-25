from click.testing import CliRunner

from dpf2.verification.standard_suite import (
    run_suite,
    standard_cases,
    verify_command,
)


def test_standard_suite_passes():
    outcomes = run_suite()
    assert set(outcomes.keys()) == {c.name for c in standard_cases()}
    assert all(result["passed"] for result in outcomes.values())


def test_verify_cli_text_and_json():
    runner = CliRunner()
    result = runner.invoke(verify_command)
    assert result.exit_code == 0
    assert "Verification suite results" in result.output

    result_json = runner.invoke(verify_command, ["--json"])
    assert result_json.exit_code == 0
    assert "brio_wu" in result_json.output
