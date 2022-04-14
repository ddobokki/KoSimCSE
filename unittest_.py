from typing import List, Type
import pandas as pd
from data.info import (
    FileFormat,
    DataPath,
    DataSource,
    TrainType,
    STSDatasetFeatures,
    UnsupervisedSimCseFeatures,
)
import re

from data.utils import get_data_path, get_folder_path

if __name__ == "__main__":
    # DataPath.ROOT,DataPath.RAW,DataSource.KLUE, TrainType.TRAIN, FileFormat.JSON
    raw_floder_path = get_folder_path(DataPath.ROOT, DataPath.RAW)
    data_path = get_data_path(
        folder_path=raw_floder_path,
        data_source=DataSource.KLUE,
        train_type=TrainType.TRAIN,
        file_format=FileFormat.JSON,
    )

    print(data_path)
    pass
