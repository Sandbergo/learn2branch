"""
Create data for tables based on test and evaluation results.
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


if __name__ == '__main__':
    results = os.listdir('eval_results/cauctions/')
    results = [pd.read_csv(f'eval_results/cauctions/{file}') for file in results]

    types = ['small', 'medium', 'big']
    for result in results:
        print(f"\npolicy: {result['policy'][0]} | problem: {result['problem'][0]}")
        for type in types:
            times = result.loc[result['type'] == type]['walltime']
            print(f'\t{type}: {round(times.mean(), 2)} +/- {round(times.std(), 2)} s')
