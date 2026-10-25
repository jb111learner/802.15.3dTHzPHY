"""独立配置解析，不修改现有 PHY 参数的默认数值。"""
from copy import deepcopy
from dataclasses import dataclass, field
from collections.abc import Mapping
from .format import FixedPointFormat


def parameter_value(params, name, default=None):
    if params is None:
        return default
    if isinstance(params, Mapping):
        return params.get(name, default)
    get = getattr(params, "get", None)
    if callable(get):
        try:
            return get(name, default)
        except (AttributeError, KeyError, TypeError):
            pass
    return getattr(params, name, default)


@dataclass(frozen=True)
class ModuleFixedPointConfig:
    enabled: bool = True
    arithmetic_level: str = "boundary"
    formats: dict = field(default_factory=dict)
    options: dict = field(default_factory=dict)

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise TypeError("module enabled must be bool")
        if self.arithmetic_level not in {"boundary", "operation"}:
            raise ValueError("arithmetic_level must be boundary or operation")

    def format(self, role="output"):
        return self.formats.get(role, self.formats["output"])

    format_for = format

    def get(self, name, default=None):
        return self.options.get(name, default)

    def to_dict(self):
        return {"enabled": self.enabled, "arithmetic_level": self.arithmetic_level,
                "formats": {k: v.to_dict() for k, v in self.formats.items()}, "options": deepcopy(self.options)}


@dataclass(frozen=True)
class FixedPointConfig:
    enable_fixed: bool = False
    default_format: FixedPointFormat = field(default_factory=FixedPointFormat)
    modules: dict = field(default_factory=dict)

    def __post_init__(self):
        if type(self.enable_fixed) is not bool:
            raise TypeError("enable_fixed must be bool")
        if not isinstance(self.default_format, FixedPointFormat):
            raise TypeError("default_format must be FixedPointFormat")
        if not isinstance(self.modules, Mapping):
            raise TypeError("modules must be a mapping")
        object.__setattr__(self, "modules", deepcopy(dict(self.modules)))

    @classmethod
    def from_dict(cls, mapping=None):
        if isinstance(mapping, cls):
            return mapping
        mapping = {} if mapping is None else mapping
        if not isinstance(mapping, Mapping):
            raise TypeError("fixed_point must be a mapping")
        unknown = set(mapping) - {"enable_fixed", "default_format", "modules"}
        if unknown:
            raise ValueError("unknown fixed_point fields: " + ", ".join(sorted(unknown)))
        if type(mapping.get("enable_fixed", False)) is not bool:
            raise TypeError("enable_fixed must be bool")
        modules = deepcopy(mapping.get("modules", {}))
        if not isinstance(modules, Mapping):
            raise TypeError("modules must be a mapping")
        result = cls(mapping.get("enable_fixed", False),
                     FixedPointFormat.from_dict(mapping.get("default_format")), modules)
        for name in modules:
            if not isinstance(name, str) or not name:
                raise TypeError("module IDs must be nonempty strings")
            result.module_config(name)
        return result

    @classmethod
    def from_params(cls, params):
        return cls.from_dict(parameter_value(params, "fixed_point", {}))

    def module_config(self, module_id):
        base = self.default_format
        def role_format(word, fractional):
            return FixedPointFormat.from_dict({"word_length": word, "fractional_length": fractional}, base)
        formats = {"input": base, "output": base, "product": role_format(48, 28),
                   "coefficient": role_format(18, 16), "phase": role_format(24, 20), "matrix": role_format(32, 16),
                   "index": role_format(32, 0), "noise": role_format(32, 24),
                   "denominator": role_format(48, 28), "weight": role_format(32, 20),
                   "accumulator": role_format(48, 28), "state": role_format(24, 12)}
        # 相关和未归一化SC变换有更大动态范围；这些是起始配置，不是性能推荐。
        if module_id in {"sc.rx.coarse_sync", "sc.rx.fine_sync", "sc.rx.channel_estimation", "sc.rx.equalizer"}:
            formats["output"] = role_format(32, 12)
        if module_id == "ldpc.llr":
            formats["output"] = role_format(8, 2)
        enabled, level, options = True, "boundary", {}
        # 父级和精确模块配置由短到长覆盖。
        matches = sorted((name for name in self.modules
                          if name == module_id or module_id.startswith(name + ".")), key=len)
        for name in matches:
            raw = self.modules[name]
            if isinstance(raw, ModuleFixedPointConfig):
                enabled, level = raw.enabled, raw.arithmetic_level
                formats.update(raw.formats)
                options.update(raw.options)
                continue
            if not isinstance(raw, Mapping):
                raise TypeError("module configuration must be a mapping: " + name)
            enabled = raw.get("enabled", enabled)
            if type(enabled) is not bool:
                raise TypeError("module enabled must be bool: " + name)
            level = raw.get("arithmetic_level", level)
            if not isinstance(raw.get("formats", {}), Mapping):
                raise TypeError("module formats must be a mapping: " + name)
            for role, value in raw.get("formats", {}).items():
                formats[role] = FixedPointFormat.from_dict(value, formats.get(role, base))
            for key, value in raw.items():
                if key.endswith("_format"):
                    role = key[:-7]
                    formats[role] = FixedPointFormat.from_dict(value, formats.get(role, base))
            if "options" in raw:
                if not isinstance(raw["options"], Mapping):
                    raise TypeError("module options must be a mapping")
                options.update(deepcopy(raw["options"]))
            options.update({k: deepcopy(v) for k, v in raw.items()
                            if k not in {"enabled", "arithmetic_level", "formats", "options"}
                            and not k.endswith("_format")})
        return ModuleFixedPointConfig(enabled, level, formats, options)

    def to_dict(self):
        def serialize(value):
            if isinstance(value, (FixedPointFormat, ModuleFixedPointConfig)):
                return value.to_dict()
            if isinstance(value, Mapping):
                return {key: serialize(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [serialize(item) for item in value]
            return deepcopy(value)
        return {"enable_fixed": self.enable_fixed, "default_format": self.default_format.to_dict(),
                # 保留显式覆盖。展开父级默认会意外覆盖子模块专用默认格式。
                "modules": {name: serialize(raw) for name, raw in self.modules.items()}}
