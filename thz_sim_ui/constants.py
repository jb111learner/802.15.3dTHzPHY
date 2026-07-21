from collections import OrderedDict

APP_NAME = '太赫兹通信物理层仿真平台'
APP_ORG = 'LiZhe'
WINDOW_MIN_SIZE = (1440, 900)

NAV_ITEMS = OrderedDict(
    [
        ('home', '首页'),
        ('link_design', '仿真链路设计'),
        ('parameter_config', '参数配置'),
        ('channel_integration', '信道集成'),
        ('batch_compare', '批量对比'),
        ('task_center', '任务中心'),
        ('result_analysis', '结果分析'),
        ('tbps_mode', '1Tbps 专项模式'),
        ('std_mode', '802.15.3d 扩展模式'),
        ('settings', '系统设置'),
    ]
)

EXTRA_PAGES = {
    'resume_recovery': '断点续跑',
}

STATUS_COLORS = {
    '运行中': '#2F6BFF',
    '已完成': '#0EA76B',
    '警告': '#E6901E',
    '已失败': '#E5484D',
    '已中断': '#E5484D',
    '草稿': '#7C869B',
    '已暂停': '#8D5CF6',
    '待恢复': '#F6A100',
}
