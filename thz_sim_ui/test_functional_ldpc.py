import json
import os
from pathlib import Path

import pytest


DEMO_PATH = Path(__file__).resolve().parent.parent / "test_cases" / "ldpc_demo.json"


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
    finally:
        page.close()
        app.processEvents()
