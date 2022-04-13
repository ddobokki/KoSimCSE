from enum import Enum

class DataSource(Enum):
    # 수정이 필요
    KLUE = 'klue-sts-v1.1_'
    KAKAO = 'sts-'

class DataPath(Enum):
    ROOT = 'data'
    RAW = 'raw'
    PREPROCESS = 'preprocess'

class DataType(Enum):
    TRAIN = 'train'
    DEV = 'dev'
    TEST = 'test'

class DataFormat(Enum):
    JSON = '.json'
    TSV = '.tsv'

class STSDatasetFeatures(Enum):
    SENTENCE1 = 'sentence1'
    SENTENCE2 = 'sentence2'