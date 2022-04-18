from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from data.info import (
    DataName,
    DataPath,
    STSDatasetFeatures,
    TrainType,
    FileFormat,
    TrainType,
    UnsupervisedSimCseFeatures,
)
from data.utils import (
    get_data_path,
    get_folder_path,
    raw_data_to_dataframe,
    make_unsupervised_sentence_data,
    wiki_preprocess,
    add_sts_df,
)

import logging

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)


"""
wiki_preprocess
"""
wiki_dataset = load_dataset("sh110495/kor-wikipedia")
#######################################################
data = []
for wiki_text in tqdm(wiki_dataset["train"]["text"]):
    wiki_text = wiki_text.replace(". ", ".\n")
    wiki_text = wiki_text.replace("\xa0", " ")
    wiki_sentences = wiki_text.split("\n")

    for wiki_sentence in wiki_sentences:
        wiki_sentence = wiki_sentence.rstrip().lstrip()
        if len(wiki_sentence) >= 10:
            data.append(wiki_sentence)
# data/utils.py로 옮기기
###########################################################
wiki_df = pd.DataFrame(data={UnsupervisedSimCseFeatures.SENTENCE.value: data})
wiki_df[UnsupervisedSimCseFeatures.SENTENCE.value] = wiki_df[
    UnsupervisedSimCseFeatures.SENTENCE.value
].apply(wiki_preprocess)


"""
sts_preprocess
"""

train_floder_path = get_folder_path(root=DataPath.ROOT, sub=DataPath.TRAIN)
dev_floder_path = get_folder_path(root=DataPath.ROOT, sub=DataPath.DEV)
test_floder_path = get_folder_path(root=DataPath.ROOT, sub=DataPath.TEST)

preprocess_wiki_data_path = get_data_path(
    folder_path=train_floder_path,
    data_source=DataName.PREPROCESS_WIKI,
    train_type=TrainType.TRAIN,
    file_format=FileFormat.CSV,
)
wiki_df = wiki_df.dropna(axis=0)
# print(len(wiki_df))
# print(wiki_df.iloc[356260])
wiki_df.to_csv(preprocess_wiki_data_path, index=False)
wiki_df = pd.read_csv(preprocess_wiki_data_path)
# print(wiki_df[wiki_df[UnsupervisedSimCseFeatures.SENTENCE.value].isna()])
logging.info(
    f"preprocess wiki train done!\nfeatures:{wiki_df.columns} \nlen: {len(wiki_df)}\nna count:{sum(wiki_df[UnsupervisedSimCseFeatures.SENTENCE.value].isna())}"
)

klue_train = raw_data_to_dataframe(
    DataPath.ROOT, DataPath.RAW, DataName.RAW_KLUE, TrainType.TRAIN, FileFormat.JSON
)

kakao_train = raw_data_to_dataframe(
    DataPath.ROOT, DataPath.RAW, DataName.RAW_KAKAO, TrainType.TRAIN, FileFormat.TSV
)

kakao_dev = raw_data_to_dataframe(
    DataPath.ROOT, DataPath.RAW, DataName.RAW_KAKAO, TrainType.DEV, FileFormat.TSV
)

kakao_test = raw_data_to_dataframe(
    DataPath.ROOT, DataPath.RAW, DataName.RAW_KAKAO, TrainType.TEST, FileFormat.TSV
)


sts_train_list = [klue_train, kakao_train]

train_sentence_list = make_unsupervised_sentence_data(sts_data_list=sts_train_list)

preprocess_sts_train_path = get_data_path(
    train_floder_path,
    data_source=DataName.PREPROCESS_STS,
    train_type=TrainType.TRAIN,
    file_format=FileFormat.CSV,
)

preprocess_sts_dev_path = get_data_path(
    dev_floder_path,
    data_source=DataName.PREPROCESS_STS,
    train_type=TrainType.DEV,
    file_format=FileFormat.TSV,
)

preprocess_sts_test_path = get_data_path(
    test_floder_path,
    data_source=DataName.PREPROCESS_STS,
    train_type=TrainType.TEST,
    file_format=FileFormat.TSV,
)


sts_train_df = pd.DataFrame(
    data={UnsupervisedSimCseFeatures.SENTENCE.value: train_sentence_list}
)
sts_train_df.dropna(axis=0).to_csv(preprocess_sts_train_path, index=False)

logging.info(
    f"preprocess sts train done!\nfeatures:{sts_train_df.columns} \nlen: {len(sts_train_df)}"
)

# sts_dev_df = kakao_dev[
#     [
#         STSDatasetFeatures.SENTENCE1.value,
#         STSDatasetFeatures.SENTENCE2.value,
#         STSDatasetFeatures.SCORE.value,
#     ]
# ]
# sts_test_df = kakao_test[
#     [
#         STSDatasetFeatures.SENTENCE1.value,
#         STSDatasetFeatures.SENTENCE2.value,
#         STSDatasetFeatures.SCORE.value,
#     ]
# ]

# sts_dev_df.to_csv(preprocess_sts_dev_path, sep="\t", index=False)

# logging.info(
#     f"preprocess sts dev done!\nfeatures:{sts_dev_df.columns} \nlen: {len(sts_dev_df)}"
# )

# sts_test_df.to_csv(preprocess_sts_test_path, sep="\t", index=False)

# logging.info(
#     f"preprocess sts test done!\nfeatures:{sts_test_df.columns} \nlen: {len(sts_test_df)}"
# )
