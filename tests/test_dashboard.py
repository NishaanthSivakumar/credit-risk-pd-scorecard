"""The dashboard runs on the committed result tables, and every control works."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


def test_dashboard_runs_without_errors():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert len(at.tabs) == 6


def test_every_control_renders():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.radio[0].set_value("intercept_slope").run()
    for option in at.selectbox[0].options:            # monitoring measure
        at.selectbox[0].set_value(option).run()
    at.select_slider[0].set_value(0.2).run()           # correlation
    at.selectbox[1].set_value("Mild recession").run()  # stress scenario
    at.selectbox[2].set_value("Max expected return").run()  # strategy
    assert not at.exception
