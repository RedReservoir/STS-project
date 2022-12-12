import pandas as pd
import re


def load_train_data():
    """
    Loads the train segment of the SemEval Task 6 dataset.

    :return: pd.DataFrame
        A pandas DataFrame with the sentence pairs and [0-1] scaled similarity scores.
    """

    # Load sentence pairs

    with open("semeval_sts_data/train/STS.input.MSRpar.txt") as MSRpar_input_file:
        MSRpar_text = MSRpar_input_file.read()

    MSRpar_ss = re.split("\n|\t", MSRpar_text)[:-1]
    MSRpar_ss_len = len(MSRpar_ss)

    MSRpar_input_df = pd.DataFrame(
        [(MSRpar_ss[i], MSRpar_ss[i + 1]) for i in range(0, MSRpar_ss_len, 2)],
        columns=["s1", "s2"]
    )

    MSRvid_input_df = pd.read_csv(
        "semeval_sts_data/train/STS.input.MSRvid.txt",
        sep='\t',
        header=None,
        names=["s1", "s2"]
    )

    SMTeuro_input_df = pd.read_csv(
        "semeval_sts_data/train/STS.input.SMTeuroparl.txt",
        sep='\t',
        header=None,
        names=["s1", "s2"]
    )

    # Load similarity scores (gold values)

    MSRpar_gs_df = pd.read_csv(
        "semeval_sts_data/train/STS.gs.MSRpar.txt",
        sep='\t',
        header=None,
        names=["gold-sim"]
    )

    MSRvid_gs_df = pd.read_csv(
        "semeval_sts_data/train/STS.gs.MSRvid.txt",
        sep='\t',
        header=None,
        names=["gold-sim"]
    )

    SMTeuro_gs_df = pd.read_csv(
        "semeval_sts_data/train/STS.gs.SMTeuroparl.txt",
        sep='\t',
        header=None,
        names=["gold-sim"]
    )

    MSRpar_gs_df /= 5
    MSRvid_gs_df /= 5
    SMTeuro_gs_df /= 5

    # Join sentence pairs and gold values

    MSRpar_full_df = pd.concat([MSRpar_input_df, MSRpar_gs_df], axis=1)
    MSRvid_full_df = pd.concat([MSRvid_input_df, MSRvid_gs_df], axis=1)
    SMTeuro_full_df = pd.concat([SMTeuro_input_df, SMTeuro_gs_df], axis=1)

    data_df = pd.concat(
        [MSRpar_full_df, MSRvid_full_df, SMTeuro_full_df],
        ignore_index=True
    )

    return data_df


def load_test_data():
    """
    Loads the test segment of the SemEval Task 6 dataset.

    :return: pd.DataFrame
        A pandas DataFrame with the sentence pairs and [0-1] scaled similarity scores.
    """

    # Load sentence pairs

    with open("semeval_sts_data/test/STS.input.MSRpar.txt") as MSRpar_input_file:
        MSRpar_text = MSRpar_input_file.read()

    MSRpar_ss = re.split("\n|\t", MSRpar_text)[:-1]
    MSRpar_ss_len = len(MSRpar_ss)

    MSRpar_input_df = pd.DataFrame(
        [(MSRpar_ss[i], MSRpar_ss[i + 1]) for i in range(0, MSRpar_ss_len, 2)],
        columns=["s1", "s2"]
    )

    MSRvid_input_df = pd.read_csv(
        "semeval_sts_data/test/STS.input.MSRvid.txt",
        sep='\t',
        header=None,
        names=["s1", "s2"]
    )

    SMTeuro_input_df = pd.read_csv(
        "semeval_sts_data/test/STS.input.SMTeuroparl.txt",
        sep='\t',
        header=None,
        names=["s1", "s2"]
    )

    # Load similarity scores (gold values)

    MSRpar_gs_df = pd.read_csv(
        "semeval_sts_data/test/STS.gs.MSRpar.txt",
        sep='\t',
        header=None,
        names=["gold-sim"]
    )

    MSRvid_gs_df = pd.read_csv(
        "semeval_sts_data/test/STS.gs.MSRvid.txt",
        sep='\t',
        header=None,
        names=["gold-sim"]
    )

    SMTeuro_gs_df = pd.read_csv(
        "semeval_sts_data/test/STS.gs.SMTeuroparl.txt",
        sep='\t',
        header=None,
        names=["gold-sim"]
    )

    MSRpar_gs_df /= 5
    MSRvid_gs_df /= 5
    SMTeuro_gs_df /= 5

    # Join sentence pairs and gold values

    MSRpar_full_df = pd.concat([MSRpar_input_df, MSRpar_gs_df], axis=1)
    MSRvid_full_df = pd.concat([MSRvid_input_df, MSRvid_gs_df], axis=1)
    SMTeuro_full_df = pd.concat([SMTeuro_input_df, SMTeuro_gs_df], axis=1)

    data_df = pd.concat(
        [MSRpar_full_df, MSRvid_full_df, SMTeuro_full_df],
        ignore_index=True
    )

    return data_df