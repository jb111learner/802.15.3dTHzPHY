import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")


def test_single_link_actual_rate_exceeds_50_gbps():
    pytest.importorskip("PySide6")
    from thz_sim_ui.services.functional_test_service import run_single_link_rate_test

    result = run_single_link_rate_test(duration=2e-7)
    assert result["ok"]
    assert result["data"]["actual_rate_bps"] >= 50e9
    assert result["data"]["actual_rate_bps"] == pytest.approx(
        result["data"]["theoretical_rate_bps"], rel=1e-12
    )
    assert {item["label"] for item in result["summary"]} >= {"理论速率", "实际速率"}
    # 结果表：实际速率位于最下方，并列出「信息比特数 ÷ 波形持续时间」公式
    metrics = [row[0] for row in result["table"]["rows"]]
    assert metrics[-1] == "实际速率"
    formula_row = next(row for row in result["table"]["rows"] if row[0] == "实际速率计算")
    assert "÷" in formula_row[1]
    assert "信息比特数 ÷ 波形持续时间" == formula_row[2]


def test_total_phy_rate_uses_required_configuration_and_exceeds_half_tbps():
    pytest.importorskip("PySide6")
    from thz_sim_ui.services.functional_test_service import run_total_phy_rate_test

    result = run_total_phy_rate_test()
    data = result["data"]
    assert result["ok"]
    assert data["configuration"] == "256QAM / 1024 子载波 / 2×2 MIMO / LDPC(11/15)"
    assert data["frame_information_bits"] == 270336
    # 与参数配置页同配置（CP 32）一致的理论速率，约 567.61 Gbps
    assert data["theoretical_rate_bps"] == pytest.approx(567614814814.8148)
    assert data["actual_rate_bps"] >= 0.5e12
    assert data["actual_rate_bps"] == pytest.approx(
        data["theoretical_rate_bps"], rel=1e-12
    )
    # 实际速率位于结果表最下方，含计算公式行；不再有 LDPC 块对齐指标
    metrics = [row[0] for row in result["table"]["rows"]]
    assert metrics[-1] == "实际速率"
    assert "LDPC 块对齐" not in metrics
    assert "空间流数" in metrics
    formula_row = next(row for row in result["table"]["rows"] if row[0] == "实际速率计算")
    assert "÷" in formula_row[1]
    assert not any("LDPC" in item["name"] for item in result["checks"])


def test_function_calibration_matches_theory_and_matlab_references():
    pytest.importorskip("PySide6")
    from thz_sim_ui.services.functional_test_service import run_function_calibration_test

    result = run_function_calibration_test(bits_per_point=10000, random_seed=2026)
    assert result["ok"]
    assert all(item["ok"] for item in result["checks"])
    # 5 条校准曲线：QPSK/16QAM/64QAM + LDPC(1440,1056) + RS(255,192)
    assert {curve["name"] for curve in result["data"]["curves"]} == {
        "QPSK", "16QAM", "64QAM", "LDPC(1440,1056)", "RS(255,192)"
    }
    assert len(result["plots"]) == 5
    assert all(plot["png"].startswith(b"\x89PNG") for plot in result["plots"])
    # 无旧校准项目表格；每张校准图下附仿真点明细表格
    assert "table" not in result
    assert len(result["tables"]) == 5
    for table in result["tables"]:
        assert table["columns"] == [
            "SNR/dB", "Eb/N0/dB", "误码数", "比特数", "实测 BER", "理论 BER"]
        assert len(table["rows"]) == 4
    # 编码曲线明细：误码数/比特数与曲线数据一致
    coded = next(curve for curve in result["data"]["curves"]
                 if curve["name"] == "LDPC(1440,1056)")
    assert coded["total_errors"][-1] < coded["total_errors"][0]
    # 编码后理论曲线：RS(255,192) 必有（MATLAB 数据缺失时回退解析式），
    # LDPC 依赖 MATLAB 参考文件，存在时按 4 点提供
    rs = next(curve for curve in result["data"]["curves"]
              if curve["name"] == "RS(255,192)")
    assert rs["coded_theory"] is not None
    assert rs["coded_theory"]["source"] in {"matlab", "analytic"}
    assert len(rs["coded_theory"]["ber"]) == 4
    ldpc_theory = coded.get("coded_theory")
    assert ldpc_theory is None or len(ldpc_theory["ber"]) == 4
    # 编码曲线的「理论 BER」列直接填充编码后理论值（0 误码点显示 —），
    # 未编码调制曲线的「理论 BER」列仍为 AWGN 理论
    uncoded_qpsk_table = result["tables"][0]
    assert uncoded_qpsk_table["rows"][0][5] == f"{result['data']['curves'][0]['theoretical'][0]:.3e}"
    for table_index, curve in zip((3, 4), (coded, rs)):
        table = result["tables"][table_index]
        theory = curve["coded_theory"]
        for row, value in zip(table["rows"], theory["ber"]):
            expected = f"{value:.3e}" if value > 1e-12 else "—"
            assert row[5] == expected


def test_rs_coded_theory_matches_matlab_official_data():
    pytest.importorskip("PySide6")
    from thz_sim_ui.services.functional_test_service import (
        _load_matlab_calibration_reference, _rs_hard_decoded_ber)

    import numpy as np

    references = _load_matlab_calibration_reference()
    rs_ref = references.get("RS(255,192)")
    assert rs_ref is not None
    analytic = _rs_hard_decoded_ber(np.asarray(rs_ref["ebn0_db"], dtype=float))
    # 解析式与 MATLAB 官方 comm.RSDecoder 数据同一量级（0 误码点跳过）
    for ebn0, matlab_ber, analytic_ber in zip(
            rs_ref["ebn0_db"], rs_ref["ber"], analytic):
        if matlab_ber == 0:
            assert analytic_ber < 1e-4
        else:
            assert analytic_ber == pytest.approx(matlab_ber, rel=0.35)
    assert np.all(np.diff(analytic) <= 0)


def test_new_ui_tests_follow_precision_in_required_order():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from thz_sim_ui.pages.functional_test_page import FunctionalTestPage, _TEST_KEYS

    app = QApplication.instance() or QApplication([])
    page = FunctionalTestPage()
    try:
        assert _TEST_KEYS[3:] == (
            "codec", "precision", "single_link_rate", "total_phy_rate",
            "function_calibration", "ber"
        )
        assert page.test_radios[5].text().startswith("单链路物理层速率测试")
        assert page.test_radios[6].text().startswith("总物理层速率测试")
        assert page.test_radios[7].text() == "校准测试"
        assert page.test_radios[8].text().startswith("误码率测试")
        assert page.param_stack.count() == 9
        assert page.result_stack.count() == 9
        assert page.total_rate_configuration.currentText().startswith("256QAM / 1024")
        page.resize(1440, 900)
        page.show()
        page.test_radios[6].setChecked(True)
        app.processEvents()
        assert page.param_stack.height() == page.param_stack.currentWidget().sizeHint().height()
        assert page.left_scroll.verticalScrollBar().maximum() == 0
        assert page.left_footer.isVisible()
        assert page.run_button.mapTo(page, page.run_button.rect().bottomLeft()).y() >= 850
    finally:
        page.close()
        app.processEvents()
