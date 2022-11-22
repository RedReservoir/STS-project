"""
Import data with pairs of sentences
"""
import pandas as pd
import re


def import_data():

    #
    # Import the pair of sentences
    #
    with open("../data/train/STS.input.MSRpar.txt") as MSRpar_input_file:
      MSRpar_text = MSRpar_input_file.read()

    MSRpar_ss = re.split("\n|\t", MSRpar_text)[:-1]
    MSRpar_ss_len = len(MSRpar_ss)

    MSRpar_input_df = pd.DataFrame(
        [(MSRpar_ss[i], MSRpar_ss[i+1]) for i in range(0, MSRpar_ss_len, 2)],
        columns=["s1", "s2"]
        )


    MSRvid_input_df = pd.read_csv(
        "../data/train/STS.input.MSRvid.txt",
        sep='\t',
        header=None,
        names=["s1", "s2"]
        )


    SMTeuro_input_df = pd.read_csv(
        "../data/train/STS.input.SMTeuroparl.txt",
        sep='\t',
        header=None,
        names=["s1", "s2"]
        )

    #
    # Import gold standard values
    #
    MSRpar_gs_df = pd.read_csv(
        "../data/train/STS.gs.MSRpar.txt",
        sep='\t',
        header=None,
        names=["gold-sim"]
        )

    MSRvid_gs_df = pd.read_csv(
        "../data/train/STS.gs.MSRvid.txt",
        sep='\t',
        header=None,
        names=["gold-sim"]
        )

    SMTeuro_gs_df = pd.read_csv(
        "../data/train/STS.gs.SMTeuroparl.txt",
        sep='\t',
        header=None,
        names=["gold-sim"]
        )

    MSRpar_gs_df /= 5
    MSRvid_gs_df /= 5
    SMTeuro_gs_df /= 5

    #
    # Join sentence pairs and gold values
    #
    MSRpar_full_df = pd.concat([MSRpar_input_df, MSRpar_gs_df], axis=1)
    MSRvid_full_df = pd.concat([MSRvid_input_df, MSRvid_gs_df], axis=1)
    SMTeuro_full_df = pd.concat([SMTeuro_input_df, SMTeuro_gs_df], axis=1)

    X_df = pd.concat(
        [MSRpar_full_df, MSRvid_full_df, SMTeuro_full_df],
        ignore_index=True
        )

    return X_df

