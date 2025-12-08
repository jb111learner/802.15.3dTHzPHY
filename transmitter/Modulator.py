import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import matplotlib as mpl
from params.PHYParams import PHYParams

# 设置中文字体和编码
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class QAMModulator:
    """
    QAM调制器类，实现MATLAB qammod函数的功能
    
    支持：
    - 比特输入模式
    - 单位平均功率归一化
    - 不同阶数的QAM调制（2^k，k为偶数）
    """
    
    def __init__(self, params, input_type='bit', unit_average_power=True):
        """
        初始化QAM调制器
        """
        self.params = params
        self.M = self.params.get("M")  # 调制阶数

        self.input_type = input_type.lower()
        self.unit_average_power = unit_average_power
        
        # 检查M是否为2的幂
        if not (self.M > 0 and (self.M & (self.M - 1)) == 0):
            raise ValueError("M必须是2的幂")
        
        # 计算每符号的比特数
        self.bits_per_symbol = int(np.log2(self.M))
        
        # 生成QAM星座图
        self.constellation = self._generate_constellation()
    
    def _generate_constellation(self):
        """生成QAM星座图"""
        # 计算星座图的维度（假设为方形QAM）
        n = int(np.sqrt(self.M))
        
        if n * n != self.M:
            raise ValueError("目前仅支持方形QAM调制（M为4, 16, 64, 256等）")
        
        # 生成星座点坐标
        x = np.arange(-(n-1), n, 2)
        y = np.arange(-(n-1), n, 2)
        X, Y = np.meshgrid(x, y)
        
        # 展平为一维数组
        constellation = X.flatten() + 1j * Y.flatten()
        
        # 如果需要单位平均功率，进行归一化
        if self.unit_average_power:
            avg_power = np.mean(np.abs(constellation) ** 2)
            constellation /= np.sqrt(avg_power)
        
        return constellation
    
    def _bits_to_symbols(self, bits):
        """将比特序列转换为符号索引"""
        # 确保比特数是每符号比特数的整数倍
        if len(bits) % self.bits_per_symbol != 0:
            raise ValueError(f"比特数必须是{self.bits_per_symbol}的整数倍")
        
        # 重塑比特数组
        bits_reshaped = bits.reshape(-1, self.bits_per_symbol)
        
        # 将每比特组转换为十进制索引
        symbols = np.zeros(len(bits_reshaped), dtype=int)
        for i in range(self.bits_per_symbol):
            symbols += bits_reshaped[:, i] * (2 ** (self.bits_per_symbol - 1 - i))
        
        return symbols
    
    def modulate(self, data):
        """
        执行QAM调制
        
        参数：
        data: 输入数据，可以是比特数组或符号索引数组
        
        返回：
        调制后的复数QAM符号
        """
        if self.input_type == 'bit':
            # 确保输入是二进制数组
            if not np.all(np.isin(data, [0, 1])):
                raise ValueError("比特输入必须只包含0和1")
            
            # 将比特转换为符号索引
            symbols = self._bits_to_symbols(data)
        elif self.input_type == 'symbol':
            # 确保符号索引有效
            if np.any(data < 0) or np.any(data >= self.M):
                raise ValueError(f"符号索引必须在0到{self.M-1}之间")
            symbols = data
        else:
            raise ValueError("input_type必须是'bit'或'symbol'")
        
        # 执行调制
        modulated = self.constellation[symbols]
        
        return modulated
    
    def plot_constellation(self, use_english=False):
        """
        绘制星座图
        
        参数：
        use_english: 是否使用英文显示标签
        """
        plt.figure(figsize=(8, 8))
        plt.scatter(self.constellation.real, self.constellation.imag, c='blue', s=100)
        
        # 添加星座点标签
        for i, (x, y) in enumerate(zip(self.constellation.real, self.constellation.imag)):
            plt.text(x, y, str(i), ha='center', va='center', color='white', fontweight='bold')
        
        plt.grid(True)
        plt.axis('equal')
        
        if use_english:
            plt.title(f'{self.M}-QAM Constellation')
            plt.xlabel('In-phase Component')
            plt.ylabel('Quadrature Component')
        else:
            plt.title(f'{self.M}-QAM 星座图')
            plt.xlabel('同相分量')
            plt.ylabel('正交分量')
        
        plt.show()


# 使用示例
if __name__ == "__main__":
    params = PHYParams()
    # 创建16-QAM调制器，比特输入，单位平均功率
    qam16 = QAMModulator(params)
    
    # 生成随机比特序列
    data_bits = np.random.randint(0, 2, 600)  # 640比特
    
    # 进行调制
    modulated_data = qam16.modulate(data_bits)
    
    print("调制后的数据（前10个符号）：")
    print(modulated_data[:10])
    
    print(f"\n星座图平均功率：{np.mean(np.abs(qam16.constellation)**2):.6f}")
    
    # 绘制星座图（使用英文避免乱码）
    qam16.plot_constellation()