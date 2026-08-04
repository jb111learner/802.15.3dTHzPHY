"""信道集成页多径参数可见性回归测试。"""

import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from thz_sim_ui.pages.channel_integration_page import ChannelIntegrationPage


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def page(app):
    return ChannelIntegrationPage()


def _assert_rows(page, visible, hidden):
    layout = page.mp_group.layout()
    for widget in visible:
        assert not widget.isHidden()
        assert not layout.labelForField(widget).isHidden()
        assert widget.isEnabled()
    for widget in hidden:
        assert widget.isHidden()
        assert layout.labelForField(widget).isHidden()


def test_simulated_custom_paths_only_show_relevant_parameters(page):
    page.mp_source.setCurrentText("仿真模型")
    page.mp_tdl_model.setCurrentText("无")

    _assert_rows(
        page,
        visible=(
            page.mp_tdl_model,
            page.mp_fading,
            page.mp_normalize,
            page.mp_paths_cfg,
        ),
        hidden=(
            page.mp_tdl_ds,
            page.mp_tdl_vel,
            page.mp_jakes,
            page.mp_measured_scenario,
            page.mp_measured_retained,
            page.mp_measured_siso_link,
            page.mp_measured_info,
        ),
    )


def test_simulated_tdl_only_show_tdl_parameters(page):
    page.mp_source.setCurrentText("仿真模型")
    page.mp_tdl_model.setCurrentText("TDL-A")

    _assert_rows(
        page,
        visible=(
            page.mp_tdl_model,
            page.mp_tdl_ds,
            page.mp_tdl_vel,
            page.mp_jakes,
        ),
        hidden=(
            page.mp_fading,
            page.mp_normalize,
            page.mp_paths_cfg,
            page.mp_measured_scenario,
            page.mp_measured_retained,
            page.mp_measured_siso_link,
            page.mp_measured_info,
        ),
    )


def test_measured_deterministic_hides_rayleigh_and_simulation_parameters(page):
    page.mp_source.setCurrentText("实测确定性回放")

    _assert_rows(
        page,
        visible=(
            page.mp_measured_scenario,
            page.mp_measured_siso_link,
            page.mp_measured_info,
        ),
        hidden=(
            page.mp_tdl_model,
            page.mp_tdl_ds,
            page.mp_tdl_vel,
            page.mp_fading,
            page.mp_normalize,
            page.mp_jakes,
            page.mp_paths_cfg,
            page.mp_measured_retained,
        ),
    )


def test_measured_rayleigh_only_show_pdp_parameters(page):
    page.mp_source.setCurrentText("实测PDP-Rayleigh")

    _assert_rows(
        page,
        visible=(
            page.mp_measured_scenario,
            page.mp_measured_retained,
            page.mp_measured_info,
        ),
        hidden=(
            page.mp_tdl_model,
            page.mp_tdl_ds,
            page.mp_tdl_vel,
            page.mp_fading,
            page.mp_normalize,
            page.mp_jakes,
            page.mp_paths_cfg,
            page.mp_measured_siso_link,
        ),
    )


def test_custom_fading_selection_controls_static_channel_flag(page):
    page.mp_source.setCurrentText("仿真模型")
    page.mp_tdl_model.setCurrentText("无")
    page.module_checks["多径模块"].setChecked(True)

    page.mp_fading.setCurrentText("frame")
    assert page.get_channel_params()["static_channel"] is False

    page.mp_fading.setCurrentText("static")
    assert page.get_channel_params()["static_channel"] is True
