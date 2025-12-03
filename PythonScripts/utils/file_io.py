import configparser
import os
import numpy as np

class FileIO():
    def __init__(self):
        # Load configuration and data
        config_path = os.path.join('.', 'config.ini')
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        py_data_path = self.config['DataSelect']['CurrentDataBase']
        self.algorithm = self.config['AlgorithmSelect']['CurrentAlgorithm']
        py_data_path = os.path.join('.', 'data', 'NpWaveData', py_data_path)
        self.datapath = py_data_path

        self.config_metadata = configparser.ConfigParser()
        self.config_metadata.read(os.path.join(py_data_path, 'Metadata.ini'))
        self.sample_rate_hz = self._load_sample_rate_hz()
        self.waveform_data = np.load(os.path.join(py_data_path, 'waveform_data.npy'))

    def join_datapath(self, path):
        return os.path.join(self.datapath, path)

    def _load_sample_rate_hz(self, default_hz=50e6):
        try:
            return self.config_metadata.getfloat('Waveform', 'SampleRateHz')
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return default_hz

    def get_metadata(self):
        grid_info = {
            'minX': int(self.config_metadata['Grid']['minX']),
            'minY': int(self.config_metadata['Grid']['minY']),
            'maxX': int(self.config_metadata['Grid']['maxX']),
            'maxY': int(self.config_metadata['Grid']['maxY']),
            'numX': int(self.config_metadata['Grid']['numX']),
            'numY': int(self.config_metadata['Grid']['numY']),
            'sampleRateHz': self.sample_rate_hz,
        }
        return grid_info
    
    def get_sample_rate_hz(self):
        return self.sample_rate_hz

    def get_waveform_data(self):
        return self.waveform_data