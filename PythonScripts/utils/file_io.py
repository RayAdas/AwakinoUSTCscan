from abc import ABC
import configparser
import os
from typing import Optional

class FileIO(ABC):
    config_path = ""
    CS_data_path = ""
    curr_CS_data_path: Optional[str] = None
    curr_CS_metadata: Optional[configparser.ConfigParser] = None
    config: configparser.ConfigParser = configparser.ConfigParser()

    rebuild_dataset_path = ""
    curr_rebuild_dataset_path: Optional[str] = None
    curr_rebuild_dataset_metadata: Optional[configparser.ConfigParser] = None

    @classmethod
    def init(cls):
        # Load configuration and data
        cls.config_path = os.path.join('.', 'config.ini')
        cls.CS_data_path = os.path.join('.', 'data', 'NpWaveData')
        cls.rebuild_dataset_path = os.path.join('.', 'data', 'RebuildDatasets')
        cls.config.read(cls.config_path)

        # !!! DEPRECATED
        try:
            cls.algorithm = cls.config['AlgorithmSelect']['CurrentAlgorithm']
        except KeyError:
            cls.algorithm = None

        # Data paths, current data path and metadata path
        try:
            db_name = cls.config['DataSelect']['CurrentDataBase']
        except KeyError:
            cls.curr_CS_data_path = None
            cls.curr_CS_metadata = None
            print("Warning: No current database selected in configuration.")
        else:
            cls.curr_CS_data_path = os.path.join(cls.CS_data_path, db_name)
            metadata_path = os.path.join(cls.curr_CS_data_path, 'Metadata.ini')

            cls.curr_CS_metadata = configparser.ConfigParser()
            if not cls.curr_CS_metadata.read(metadata_path):
                cls.curr_CS_metadata = None
                print("Warning: Metadata.ini not found in the current database path.")

        # rebuild dataset
        try:
            rebuild_dataset_name = cls.config['DataSelect']['CurrentRebuildDataset']
        except KeyError:
            cls.curr_rebuild_dataset_path = None
            cls.curr_rebuild_dataset_metadata = None
            print("Warning: No current rebuild dataset selected in configuration.")
        else:
            cls.curr_rebuild_dataset_path = os.path.join(cls.rebuild_dataset_path, rebuild_dataset_name)
            rebuild_metadata_path = os.path.join(cls.curr_rebuild_dataset_path, 'Metadata.ini')

            cls.curr_rebuild_dataset_metadata = configparser.ConfigParser()
            if not cls.curr_rebuild_dataset_metadata.read(rebuild_metadata_path):
                cls.curr_rebuild_dataset_metadata = None
                print("Warning: Metadata.ini not found in the current rebuild dataset path.")
