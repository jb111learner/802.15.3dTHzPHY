from pathlib import Path
import json

import numpy as np

from thz_sim_ui.services import backend
from thz_sim_ui.data.mock_data import TaskItem
from simulation.SimulationManager import SimulationManager
from params.PHYParams import PHYParams
from transmitter.Pulseshaper import TxPulseShaper


class _RecordingConnection:
    def __init__(self):
        self.messages = []
        self.closed = False

    def send(self, message):
        self.messages.append(message)

    def close(self):
        self.closed = True


def test_batch_point_requires_minimum_independent_runs_before_error_stop():
    should_continue = backend._should_continue_batch_point

    assert should_continue(5000, 1, 5000, 20, 500)
    assert not should_continue(5000, 20, 5000, 20, 500)
    assert should_continue(4999, 20, 5000, 20, 500)
    assert not should_continue(4999, 500, 5000, 20, 500)


def test_ofdm_unit_oversampling_uses_identity_filter():
    params = PHYParams()
    params.update(link_mode='ofdm', oversampling=1)

    pulse_shaper = TxPulseShaper(params)

    assert np.array_equal(pulse_shaper._design_lowpass_filter(), np.array([1.0]))


def test_link_design_mode_has_priority_and_measured_channel_does_not_override_it():
    mapped = backend.BackendService.map_ui_params_to_phy_params({
        '链路模式分区': '单载波模式',
        '波形类型': '多载波OFDM',
        '_channel_params': {
            'enable_multipath': True,
            'multipath_source': 'measured',
            'measured_channel_scenario': '50cm',
        },
    })

    assert mapped['link_mode'] == 'sc-fde'
    assert mapped['sample_rate'] == 30e9


def test_batch_worker_uses_linked_project_instead_of_result_label(monkeypatch, tmp_path):
    loaded_projects = []
    connection = _RecordingConnection()

    def fake_load(project_name):
        loaded_projects.append(project_name)
        return {'工程名称': project_name}

    def fake_execute(
        ui_params, snr, max_frames, min_independent_runs, min_errors,
        progress_callback=None,
    ):
        assert min_independent_runs == 1
        if progress_callback:
            progress_callback(1, 0)
        return {
            'SNRdB': snr,
            'BER': 0.1,
            'total_bits': 10,
            'total_errors': 1,
            'total_frames': 1,
        }

    monkeypatch.setattr(backend, '_load_batch_project_config', fake_load)
    monkeypatch.setattr(backend, '_execute_batch_point', fake_execute)
    monkeypatch.setattr(
        backend,
        '_save_batch_compare_chart',
        lambda results, chart_path: Path(chart_path).write_bytes(b'png'),
    )

    backend._batch_compare_process_entry(
        connection,
        {
            '配置方案': [{'工程': 'Project4', '结果标签': '论文曲线A'}],
            'SNR最小值': 10.0,
            'SNR最大值': 10.0,
            'SNR步长': 1.0,
            '每点最大帧数': 1,
            '每点最少独立运行次数': 1,
            '每点最少错误比特': 1,
        },
        str(tmp_path),
    )

    assert loaded_projects == ['Project4']
    assert connection.closed
    assert connection.messages[-1][0] == 'finished'
    assert [message[0] for message in connection.messages].count('progress') == 2
    assert (tmp_path / '论文曲线A' / 'scheme_results.json').exists()
    assert (tmp_path / 'compare_results' / 'ber_compare.png').exists()


def test_batch_worker_persists_each_completed_point_before_later_interruption(
    monkeypatch, tmp_path,
):
    connection = _RecordingConnection()
    executed_snrs = []

    monkeypatch.setattr(backend, '_load_batch_project_config', lambda _name: {})

    def fake_execute(
        ui_params, snr, max_frames, min_independent_runs, min_errors,
        progress_callback=None,
    ):
        executed_snrs.append(snr)
        if len(executed_snrs) == 2:
            raise KeyboardInterrupt('simulated interruption')
        return {'SNRdB': snr, 'BER': 0.1}

    monkeypatch.setattr(backend, '_execute_batch_point', fake_execute)

    backend._batch_compare_process_entry(
        connection,
        {
            '配置方案': [{'工程': 'Project4', '结果标签': '方案A'}],
            'SNR最小值': 10.0,
            'SNR最大值': 11.0,
            'SNR步长': 1.0,
            '每点最大帧数': 1,
            '每点最少独立运行次数': 1,
            '每点最少错误比特': 1,
        },
        str(tmp_path),
    )

    result_path = tmp_path / '方案A' / 'scheme_results.json'
    assert json.loads(result_path.read_text(encoding='utf-8')) == [
        {'snr': 10.0, 'result': {'SNRdB': 10.0, 'BER': 0.1}},
    ]
    assert connection.messages[-1][0] == 'fatal_error'
    assert connection.closed


def test_running_task_elapsed_refreshes_without_progress_event(monkeypatch):
    task = TaskItem(
        name='running', project='project', mode='批量对比', tags='', stage='仿真',
        progress=25, elapsed='00:00:00', eta='未知', status='运行中', start_time=100.0,
    )
    monkeypatch.setattr(backend, 'RECENT_TASKS', [task])
    monkeypatch.setattr(backend.time, 'time', lambda: 165.0)
    service = backend.BackendService()
    updates = []
    service.tasks_updated.connect(updates.append)

    service._refresh_running_task_elapsed()

    assert task.elapsed == '00:01:05'
    assert task.eta == '00:03:15'
    assert updates and updates[-1][0] is task


def test_single_run_reports_pipeline_stage_progress(monkeypatch, tmp_path):
    manager = SimulationManager(
        base_params=PHYParams(), output_dir=str(tmp_path), save_plots=False,
    )
    manager.set_control_params({'run_times': 1})

    def fake_run_once(override_params=None, progress_callback=None):
        for progress in (5, 35, 55, 90, 98):
            progress_callback(progress)
        return {'metrics': {}}

    monkeypatch.setattr(manager, 'run_once', fake_run_once)
    reported = []

    manager.run_monte_carlo(progress_callback=reported.append)

    assert reported == [0, 5, 35, 55, 90, 98, 100]
