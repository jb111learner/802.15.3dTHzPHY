"""现有模块的轻量接入接口，关闭定点时不改变算法。"""
from dataclasses import replace
from .config import FixedPointConfig
from .context import FixedPointContext


class FixedPointMixin:
    def _init_fixed(self, params=None, module_id=None):
        self.fixed_module_id = module_id or type(self).__name__
        self._fixed_context = FixedPointContext.from_params(params)

    @property
    def fixed_context(self):
        context = getattr(self, "_fixed_context", None)
        if context is None:
            self._init_fixed(getattr(self, "params", None), getattr(self, "fixed_module_id", type(self).__name__))
            context = self._fixed_context
        return context

    @property
    def enable_fixed(self):
        return self.fixed_context.module_enabled(getattr(self, "fixed_module_id", type(self).__name__))

    def configure_fixed(self, enable_fixed, fixed_config=None, fixed_context=None):
        if type(enable_fixed) is not bool:
            raise TypeError("enable_fixed must be bool")
        if fixed_context is not None:
            if not isinstance(fixed_context, FixedPointContext):
                raise TypeError("fixed_context must be FixedPointContext")
            if fixed_context.config.enable_fixed != enable_fixed:
                raise ValueError("configure_fixed switch disagrees with supplied context")
            self._fixed_context = fixed_context
        else:
            current = self.fixed_context
            cfg = FixedPointConfig.from_dict(fixed_config) if fixed_config is not None else current.config
            self._fixed_context = FixedPointContext(replace(cfg, enable_fixed=enable_fixed), {
                "enabled": current.range_enabled, "reservoir_size": current.range_analyzer.reservoir_size})
        if not hasattr(self, "fixed_module_id"):
            self.fixed_module_id = type(self).__name__
        return self

    def _fq(self, value, role="output", node=None):
        return self.fixed_context.quantize(value, self.fixed_module_id, role, node)
