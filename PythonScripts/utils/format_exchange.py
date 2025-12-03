"""
@discription: 读取USTCscan产生的txt文件, 转换为numpy格式
"""

import os
import configparser
import numpy as np


class WaveformLoader:
    """Strategy interface for loading waveform data."""

    def load(self, file_path: str, signal_length: int) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _normalize_waveform(data: np.ndarray, signal_length: int) -> np.ndarray:
        data = data.flatten()
        if len(data) < signal_length:
            data = np.pad(data, (0, signal_length - len(data)), "constant")
        else:
            data = data[:signal_length]
        return data


class NpyWaveformLoader(WaveformLoader):
    def load(self, file_path: str, signal_length: int) -> np.ndarray:
        data = np.load(file_path)
        return self._normalize_waveform(data, signal_length)


class TxtWaveformLoader(WaveformLoader):
    def load(self, file_path: str, signal_length: int) -> np.ndarray:
        data = np.loadtxt(file_path, delimiter=",")
        return self._normalize_waveform(data, signal_length)


# Strategy mapping keyed by file extension.
WAVEFORM_LOADER_REGISTRY = {
    ".npy": NpyWaveformLoader(),
    ".txt": TxtWaveformLoader(),
}

config_path = os.path.join('.', 'config.ini')
config = configparser.ConfigParser()
config.read(config_path)

src_data_path = config['DataSelect']['CurrentDataBase']
src_data_path = os.path.join('.', 'data', 'OSCget', src_data_path)
py_data_path = config['DataSelect']['CurrentDataBase']
py_data_path = os.path.join('.', 'data', 'NpWaveData', py_data_path)

# 解析index.ini文件
index_path = os.path.join(src_data_path, 'index.ini')
config = configparser.ConfigParser()
config.read(index_path)

# 读取网格信息
grid_info = {
    'minX': int(config['Grid']['minX']),
    'minY': int(config['Grid']['minY']),
    'maxX': int(config['Grid']['maxX']),
    'maxY': int(config['Grid']['maxY']),
    'numX': int(config['Grid']['numX']),
    'numY': int(config['Grid']['numY']),
}

wave_info = {
    'signal_length': 1000,
    'sampleRate': 100e6,
    'head2trigger': 0.0,
}

# 检查[Wave]参数是否存在，若存在则使用该值
if 'Wave' in config and 'signalLength' in config['Wave']:
    wave_info['signal_length'] = int(float(config['Wave']['signalLength']))
if 'Wave' in config and 'sampleRate' in config['Wave']:
    wave_info['sampleRate'] = float(config['Wave']['sampleRate'])
if 'Wave' in config and 'head2trigger' in config['Wave']:
    wave_info['head2trigger'] = float(config['Wave']['head2trigger'])

# 初始化存储波形数据的numpy数组
waveform_data = np.zeros((grid_info['numY'], grid_info['numX'], wave_info['signal_length']), dtype=float)

# 遍历每条数据并读取波形数据
for y in range(grid_info['numY']):
    section = str(y)
    if section in config:
        for x in range(grid_info['numX']):
            signal_length = wave_info['signal_length']
            if str(x) not in config[section]:
                waveform_data[y, x] = np.zeros(signal_length, dtype=float)
                continue
            file_path = config[section][str(x)]
            file_path = os.path.join(src_data_path, file_path)
            extension = os.path.splitext(file_path)[1].lower()
            loader = WAVEFORM_LOADER_REGISTRY.get(extension)
            if loader is None:
                waveform_data[y, x] = np.zeros(signal_length, dtype=float)
                continue
            try:
                waveform_data[y, x] = loader.load(file_path, signal_length)
            except (OSError, ValueError):
                waveform_data[y, x] = np.zeros(signal_length, dtype=float)

if not os.path.exists(py_data_path):
    os.makedirs(py_data_path)

# 保存数据为npz
np.savez_compressed(
    os.path.join(py_data_path, 'data.npz'),
    waveform_data=waveform_data,
    grid_info=grid_info,
    wave_info=wave_info
)