from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class TaskItem:
    name: str
    project: str
    mode: str
    tags: str
    stage: str
    progress: int
    elapsed: str
    eta: str
    status: str
    is_current: bool = False
    start_time: float = 0.0
    result: dict | None = None
    result_path: str | None = None


RECENT_TASKS: List[TaskItem] = [
    # TaskItem('THz-Link-AWGN-001', '演示工程-A', '单载波 / 浮点', 'QPSK, LDPC', '均衡', 68, '00:14:20', '00:06:00', '运行中'),
    # TaskItem('THz-MIMO-Scan-014', 'MIMO 对比项目', '2x2 MIMO', '16QAM, MMSE', '参数扫描', 100, '01:12:08', '-', '已完成'),
    # TaskItem('1Tbps-Proposal-003', '高速链路论证', '1Tbps 新波形', '64QAM, Polar', '速率评估', 45, '00:08:10', '00:10:40', '已暂停'),
    # TaskItem('STD-802153d-002', '标准模式验证', '802.15.3d', '模板-B', '信道估计', 57, '00:23:54', '00:08:22', '待恢复'),
]

CAPABILITIES = [
    '单载波', '多载波', '定点', '浮点', '非 MIMO', '2×2 MIMO', '802.15.3d THz', '1Tbps 专项链路', '批量仿真', '断点续跑'
]

RECENT_RESULTS = [
    'BER 曲线', '星座图', '频谱图', '信道估计图', '方案对比', '导出报告'
]

LINK_FLOW = [
    '比特源', '编码', '扰码', '调制', '帧构建', '波形成形', '信道', '同步', '频偏补偿', '信道估计', '均衡', '解调', '译码', '结果统计'
]

CHANNEL_MODULES = [
    'AWGN 模块', '多径模块', 'CFO 模块', '相位噪声模块', 'THz 吸收模块', '天线与波束模块', 'MIMO 信道模块', '自定义信道模块'
]

RESULT_CHARTS = [
    'BER 曲线', 'BLER 曲线', '吞吐率曲线', '频谱效率曲线', 'NMSE 曲线', '时延统计图'
]
