import json
import os
from pathlib import Path

import pytest


DEMO_PATH = Path(__file__).resolve().parent.parent / "test_cases" / "ldpc_demo.json"
RS_DEMO_PATH = Path(__file__).resolve().parent.parent / "test_cases" / "rs_demo.json"


def test_rs_demo_contains_complete_matlab_references():
    config = json.loads(RS_DEMO_PATH.read_text(encoding="utf-8"))
    n = config["rs_params"]["n"]
    k = config["rs_params"]["k"]
    for case in config["cases"]:
        reference = case["matlab_reference"]
        packet_count = (len(case["input"]) + k - 1) // k
        assert len(reference["encoded"]) == packet_count * n
        assert reference["decoded"] == case["input"]


def test_rs_demo_matches_python_codec(monkeypatch):
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    pytest.importorskip("PySide6")
    from thz_sim_ui.services.functional_test_service import run_codec_test

    result = run_codec_test(RS_DEMO_PATH.read_text(encoding="utf-8"))
    assert result["ok"]
    assert all(
        case["matlab_comparison"] == "编码一致；译码一致"
        for case in result["case_results"]
    )


def test_ldpc_demo_declared_lengths_and_block_counts():
    pytest.importorskip("PySide6")
    from thz_sim_ui.services.functional_test_service import run_codec_test

    config_text = DEMO_PATH.read_text(encoding="utf-8")
    config = json.loads(config_text)
    assert [len(bytes.fromhex(case["input_hex"])) for case in config["cases"]] == [168, 300]

    result = run_codec_test(config_text)
    assert result["ok"]
    assert "1 个码字块" in result["detail_lines"][0]
    assert "2 个码字块" in result["detail_lines"][1]
    assert result["table"]["columns"][3] == "LDPC校验位(hex)"
    assert result["table"]["rows"][0][1] != result["table"]["rows"][0][2]
    assert result["table"]["rows"][0][3].startswith("块1: ")
    assert "块2: " in result["table"]["rows"][1][3]
    assert len(result["case_results"]) == 2
    assert result["case_results"][0]["input_size"] == "1344 bit"
    assert result["case_results"][0]["block_count"] == 1
    assert result["case_results"][1]["block_count"] == 2
    assert result["case_results"][0]["decoded_data"] == config["cases"][0]["input_hex"]
    assert result["case_results"][0]["matlab_encoded"] == config["cases"][0]["matlab_reference"]["encoded"]
    assert result["case_results"][0]["matlab_decoded"] == config["cases"][0]["matlab_reference"]["decoded"]
    assert result["case_results"][0]["matlab_comparison"] == "编码一致；译码一致"
    assert result["case_results"][1]["matlab_comparison"] == "编码一致；译码一致"


def test_offline_matlab_reference_comparison():
    pytest.importorskip("PySide6")
    from thz_sim_ui.services.functional_test_service import _matlab_reference_comparison

    rs_case = {"matlab_reference": {"encoded": [1, 2, 3], "decoded": [1, 2]}}
    rs_result = _matlab_reference_comparison(rs_case, "[1, 2, 3]", "[1, 2]", "gf")
    assert rs_result["matlab_comparison"] == "编码一致；译码一致"

    ldpc_case = {"matlab_reference": {"encoded": "AA BB", "decoded": "CC"}}
    ldpc_result = _matlab_reference_comparison(ldpc_case, "AABB", "CD", "hex")
    assert ldpc_result["matlab_comparison"] == "编码一致；译码不一致"


def test_codec_selector_loads_matching_ldpc_demo():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from thz_sim_ui.pages.functional_test_page import FunctionalTestPage

    app = QApplication.instance() or QApplication([])
    page = FunctionalTestPage()
    try:
        assert json.loads(page.codec_editor.toPlainText())["code_type"] == "RS"
        page.codec_type_combo.setCurrentIndex(1)
        assert json.loads(page.codec_editor.toPlainText())["code_type"] == "LDPC"
        assert page.codec_param_stack.currentIndex() == 1
        assert not page.codec_case_box.isVisible()

        page.codec_preset_combo.setCurrentText("自定义")
        config = page._codec_config_from_form()
        assert config["code_type"] == "LDPC"
        assert config["ldpc_params"]["rate"] == "14/15"
        assert len(bytes.fromhex(config["cases"][0]["input_hex"])) == 168

        page.codec_type_combo.setCurrentIndex(0)
        page.codec_preset_combo.setCurrentText("单点错误")
        config = page._codec_config_from_form()
        assert config["code_type"] == "RS"
        assert config["cases"][0]["errors"] == [{"pos": 0, "value": 1}]
    finally:
        page.close()
        app.processEvents()
