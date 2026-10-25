"""界面/项目数值配置的纯函数映射，不依赖 Qt。"""
from copy import deepcopy


def map_numeric_ui_parameters(ui_params):
    keys = {"仿真数值模式", "定点字长", "定点小数位", "定点舍入", "定点饱和",
            "范围统计", "_fixed_point", "fixed_point", "range_analysis"}
    if not keys.intersection(ui_params):
        return {}
    from fixedpoint.config import FixedPointConfig
    config = deepcopy(ui_params.get("_fixed_point", ui_params.get("fixed_point", {})))
    if "仿真数值模式" in ui_params:
        mode = str(ui_params["仿真数值模式"])
        if mode not in {"浮点", "定点"}:
            raise ValueError("仿真数值模式仅支持浮点或定点")
        config["enable_fixed"] = mode == "定点"
    fmt = config.setdefault("default_format", {})
    for ui_key, field in (("定点字长", "word_length"), ("定点小数位", "fractional_length")):
        if ui_key in ui_params:
            fmt[field] = int(ui_params[ui_key])
    if "定点舍入" in ui_params:
        fmt["rounding"] = str(ui_params["定点舍入"])
    if "定点饱和" in ui_params:
        fmt["saturation"] = str(ui_params["定点饱和"]) == "启用"
    FixedPointConfig.from_dict(config)
    result = {"fixed_point": config}
    if "range_analysis" in ui_params:
        result["range_analysis"] = deepcopy(ui_params["range_analysis"])
    if "范围统计" in ui_params:
        result.setdefault("range_analysis", {})["enabled"] = str(ui_params["范围统计"]) == "启用"
    return result
