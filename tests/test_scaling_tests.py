from scripts.scaling_tests import weak_scaling, strong_scaling, document_results

def test_scaling_functions(tmp_path):
    weak = weak_scaling([1, 2, 4])
    strong = strong_scaling([1, 2, 4])
    assert weak[1] == 1.0 and strong[4] == 0.25
    report = tmp_path / "report.md"
    document_results(report, weak, strong)
    assert report.exists()
