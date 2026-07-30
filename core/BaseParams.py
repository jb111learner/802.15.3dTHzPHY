import numpy as np

class BaseParams:
    """
    参数基类：定义参数管理的统一接口
    子类需重写：_init_params() 初始化具体参数
              validate() 校验参数合法性
    """
    def __init__(self):
        self._params = {}  # 存储所有参数的字典
        self._init_params()  # 子类初始化具体参数
        self.validate()      # 校验参数合法性

    def _init_params(self):
        """初始化参数（子类必须重写）"""
        raise NotImplementedError("子类必须实现 _init_params() 方法")

    def validate(self):
        """校验参数合法性（子类必须重写）"""
        raise NotImplementedError("子类必须实现 validate() 方法")

    def get(self, key, default=None):
        """获取参数值，支持默认值"""
        return self._params.get(key, default)

    def update(self, **kwargs):
        """动态更新参数（支持批量更新）"""
        for key, value in kwargs.items():
            if key not in self._params:
                raise KeyError(f"参数 {key} 不存在")
            self._params[key] = value
        self.validate()  # 更新后重新校验

    def clone(self):
        """深拷贝参数对象并返回。"""
        new_obj = self.__class__()
        new_obj._params = self._params.copy()
        new_obj.validate()
        return new_obj

    def __str__(self):
        """打印所有参数"""
        return "\n".join([f"{key}: {value}" for key, value in self._params.items()])

# # 使用示例（子类实现）
# class PHYParams(BaseParams):
#     def _init_params(self):
#         self._params = {
#             "fs": 1760e6,          # 码片速率
#             "M": 4,                # 调制阶数（4-QAM）
#             "SNRdB": 10,           # 信噪比
#             "N": 64,               # FFT点数
#             "chan_delays": np.array([0, 4, 8, 12]),  # 多径时延
#             "chan_gains": np.array([0.9, 0.7, 0.5*np.exp(-1j*np.pi/6), 0.25*np.exp(-1j*np.pi/3)]),  # 多径增益
#         }

#     def validate(self):
#         """校验参数合法性"""
#         assert self.get("SNRdB") >= 0, "SNR不能为负"
#         assert self.get("N") == 64, "当前仅支持64点FFT"
#         assert len(self.get("chan_delays")) == len(self.get("chan_gains")), "多径时延和增益长度必须一致"

# # 测试
# if __name__ == "__main__":
#     params = PHYParams()
#     print(params)
#     params.update(SNRdB=15)
#     print("\n更新后：")
#     print(params)