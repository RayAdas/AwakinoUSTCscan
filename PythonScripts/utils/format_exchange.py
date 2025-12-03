"""
@discription: 读取USTCscan产生的txt文件, 转换为numpy格式
"""

import os
import configparser
import numpy as np

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

signal_length = 1000
# 检查[Wave][signalLength]是否存在，若存在则使用该值
if 'Wave' in config and 'signalLength' in config['Wave']:
    signal_length = int(float(config['Wave']['signalLength']))

# 初始化存储波形数据的numpy数组
waveform_data = np.zeros((grid_info['numY'], grid_info['numX'], signal_length), dtype=float)

# 遍历每条数据并读取波形数据
for y in range(grid_info['numY']):
    section = str(y)
    if section in config:
        for x in range(grid_info['numX']):
            if str(x) not in config[section]:
                waveform_data[y, x] = np.zeros(signal_length, dtype=float)
                continue
            file_path = config[section][str(x)]
            file_path = os.path.join(src_data_path, file_path)
            
            if file_path.endswith('.npy'):
                w = np.load(file_path)
                w = w.flatten()
                if len(w) < signal_length:
                    w = np.pad(w, (0, signal_length - len(w)), 'constant')
                else:
                    w = w[:signal_length]
                waveform_data[y, x] = w
            elif file_path.endswith('.txt'):
                # 使用 np.loadtxt 读取波形数据文件
                w = np.loadtxt(file_path, delimiter=',')
                w = w.flatten()
                if len(w) < signal_length:
                    w = np.pad(w, (0, signal_length - len(w)), 'constant')
                else:
                    w = w[:signal_length]
                waveform_data[y, x] = w

if not os.path.exists(py_data_path):
    os.makedirs(py_data_path)
np.save(os.path.join(py_data_path, 'waveform_data.npy'), waveform_data)

config_out = configparser.ConfigParser()
config_out.add_section('Grid')
config_out.set('Grid','minX',config['Grid']['minX'])
config_out.set('Grid','minY',config['Grid']['minY'])
config_out.set('Grid','maxX',config['Grid']['maxX'])
config_out.set('Grid','maxY',config['Grid']['maxY'])
config_out.set('Grid','numX',config['Grid']['numX'])
config_out.set('Grid','numY',config['Grid']['numY'])

config_out.add_section('Wave')
if 'Wave' in config and 'signalLength' in config['Wave']:
    config_out.set('Wave', 'signalLength', str(signal_length))
if 'Wave' in config and 'sampleRate' in config['Wave']:
    config_out.set('Wave', 'sampleRate', config['Wave']['sampleRate'])
if 'Wave' in config and 'head2trigger' in config['Wave']:
    config_out.set('Wave', 'head2trigger', config['Wave']['head2trigger'])

with open(os.path.join(py_data_path, 'Metadata.ini'), 'w') as configfile:
    config_out.write(configfile)