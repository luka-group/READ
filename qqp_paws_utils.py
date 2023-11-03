
import pandas as pd
from datasets import Dataset


def load_qqp_paws_dataset(preprocess_function, data_args):
    """
    Load and preprocess qqp paws dataset
    """
    qqp_paws_df = pd.read_table('./dataset/qqp_paws/dev_and_test.tsv')

    return process_qqp_paws_dataset(preprocess_function, qqp_paws_df, data_args)


def process_qqp_paws_dataset(preprocess_function, dataframe, data_args):
    """
    Util function of load_qqp_paws_dataset function
    """
    dataset = Dataset.from_pandas(
        dataframe.rename(columns={"sentence1": "question1", "sentence2": "question2"}))
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on qqp_paws dataset",
    )
    return dataset
