import datasets
import pandas as pd
import os
from data.info import (
    DataSource,
    DataPath,
    DataType,
    DataFormat,
    STSDatasetFeatures
)

klue_path = os.path.join(
    DataPath.ROOT.value
)
klue_sts_train = pd.read_json()