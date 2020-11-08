"""
Analyze features of the generated problems.
"""
import os
import sys
import importlib
import gzip
import argparse
import csv
import math
import numpy as np
import pandas as pd
import time
import pickle
import pathlib
import torch

# import utilities
from utilities_mlp import MLPDataset as Dataset
from utilities_mlp import load_batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--problem',
        help='MILP instance type to process.',
        type=str,
        default='cauctions',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '--data_path',
        help='name of the folder where train and valid folders are present. Assumes `data/samples` as default.',
        type=str,
        default="data/samples",
    )
    args = parser.parse_args()
    problem_folders = {
        'setcover': '500r_1000c_0.05d',
        'cauctions': '100_500',
        'facilities': '100_100_5',
        'indset': '750_4',
    }
    problem_folder = problem_folders[args.problem]
    device = torch.device("cpu")
    rng = np.random.RandomState(101)

    test_files = list(pathlib.Path(f"{args.data_path}/{args.problem}/{problem_folder}/test").glob('sample_node_135_57.pkl'))
    # test_files = list(pathlib.Path(f"{args.data_path}/{args.problem}/{problem_folder}/test").glob('sample_*0.pkl'))
    test_files = [str(x) for x in test_files]
    # print(len(test_files))

    with gzip.open(test_files[0], 'rb') as f:
        sample = pickle.load(f)

    obss, target, obss_feats, _ = sample['obss']
    #print(obss)
    
    v, _, _ = obss
    print(v.shape)
    sample_cand_scores = obss_feats['scores']
    sample_cands = np.where(sample_cand_scores != -1)[0]

    v_feats = v[sample_cands]
    # v_feats = utilities._preprocess(v_feats, mode='min-max-2')

    exit(0)

    chosen_test_files = rng.choice(test_files, 1, replace=True)

    test_data = Dataset(chosen_test_files)

    test_data = torch.utils.data.DataLoader(
        test_data, batch_size=1,
        shuffle=False, num_workers=0, collate_fn=load_batch)

    for batch in test_data:
        (cand_features, n_cands, best_cands, cand_scores, weights) = batch
        # cand_features, n_cands, best_cands, cand_scores, weights = map(lambda x:x.to(device), batch)
        print(cand_features.shape)
        # print(n_cands)
        # print(best_cands)
        # print(cand_scores[338])
        # print(weights)
        # torch.set_printoptions(profile="full")
        # print(cand_features) # prints the whole tensor
        # exit(0)
