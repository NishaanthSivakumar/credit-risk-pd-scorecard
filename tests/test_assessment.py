"""Decision rules for the validation outcome."""
from pdmodel.assessment import Dimension, decide

KEYS = ["conceptual", "data", "target", "discrimination", "calibration", "stability", "oot",
        "challenger", "sensitivity", "limitations"]


def _dims(**status):
    return [Dimension(k, k, status.get(k, "GREEN"), "") for k in KEYS]


def test_all_green_is_approve():
    assert decide(_dims())[0] == "Approve"


def test_data_problem_is_reject_even_if_calibration_fails():
    assert decide(_dims(data="RED", calibration="RED"))[0] == "Reject"


def test_unstable_ranking_is_redevelop():
    assert decide(_dims(stability="RED", calibration="RED"))[0] == "Redevelop"


def test_level_problem_only_is_recalibrate():
    assert decide(_dims(calibration="RED", discrimination="AMBER"))[0] == "Recalibrate"


def test_weak_discrimination_depends_on_challenger():
    assert decide(_dims(discrimination="RED"))[0] == "Conditionally approve"
    assert decide(_dims(discrimination="RED", challenger="AMBER"))[0] == "Redevelop"


def test_amber_is_conditional():
    assert decide(_dims(stability="AMBER", limitations="AMBER"))[0] == "Conditionally approve"
    assert decide(_dims(limitations="AMBER"))[0] == "Approve"
