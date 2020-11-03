"""
Analyze features of the generated problems.
"""
import os
import sys
import importlib
import argparse
import csv
import math
import numpy as np
import pandas as pd
import time
import pickle
import pathlib

import utilities
from utilities_mlp import MLPDataset as Dataset
from utilities_mlp import load_batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        default='cauctions',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    args = parser.parse_args()
    problem_folders = {
        'setcover': '500r_1000c_0.05d',
        'cauctions': '100_500',
        'facilities': '100_100_5',
        'indset': '750_4',
    }

    for batch in dataloader:
        cand_features, n_cands, best_cands, cand_scores, weights = map(lambda x:x.to(device), batch)

