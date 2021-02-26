"""
Create data for tables based on test and evaluation results.
"""
import pretty_errors
import os
import sys
import importlib
import argparse
import csv
import numpy as np
import time
import pickle
import pathlib
import gzip

import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-m', '--model_string',
        help='searches for this string in respective trained_models folder',
        type=str,
        default='',
    )
    parser.add_argument(
        '--model_name',
        help='searches for this model_name in respective trained_models folder',
        type=str,
        default='',
    )
    parser.add_argument(
        '--test_path',
        help='if given, searches for samples in this path',
        type=str,
        default='',
    )
    args = parser.parse_args()

    ### HYPER PARAMETERS ###
    seeds = [61]  # [0, 61, 70]  #  [35]  # TODO: [0, 1, 2]
    test_batch_size = 128
    num_workers = 0  # TODO: 5

    problem_folders = {
        'setcover': '500r_1000c_0.05d',
        'cauctions': '100_500',
        'facilities': '100_100_5',
        'indset': '750_4',
    }

    problem_folder = problem_folders[args.problem]
    resultdir = "test_results"
    os.makedirs(resultdir, exist_ok=True)
    result_file = f"{resultdir}/{args.problem}_test_{time.strftime('%Y%m%d-%H%M%S')}.csv"

    ### MODELS TO TEST ###
    if args.model_string != "":
        models_to_test = [y for y in pathlib.Path(f"trained_models/{args.problem}").iterdir() if args.model_string in y.name and 'mlp_' in y.name]
        assert len(models_to_test) > 0, f"no model matched the model_string: {args.model_string}"
    elif args.model_name != "":
        model_path = pathlib.Path(f"trained_models/{args.problem}/{args.model_name}")
        assert model_path.exists(), f"path: {model_path} doesn't exist"
        assert 'mlp_' in model_path.name, f"only tests mlp models. model_path doesn't look like its mlp: {model_path}"
        models_to_test = [model_path]
    else:
        models_to_test = [y for y in pathlib.Path(f"trained_models/{args.problem}").iterdir() if 'mlp_' in y.name]
        assert len(models_to_test) > 0, f"no model matched the model_string: {args.model_string}"


    ### NUMPY / TORCH SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = torch.device("cpu")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    ### SET-UP DATASET ###
    problem_folder = f"data/samples/{args.problem}/{problem_folders[args.problem]}/test"
    if args.test_path:
        problem_folder = args.test_path

    evaluated_policies = [['mlp', model_path] for model_path in models_to_test]


    for model_type, model_path in evaluated_policies:
        print(f"{model_type}:{model_path.name}...")
        for seed in seeds:

            policy = {}
            policy['name'] = model_path.name
            policy['type'] = model_type

            # load model
            best_params = str(model_path / f"{seed}/best_params.pkl")
            sys.path.insert(0, os.path.abspath(f"models/mlp"))
            import model
            importlib.reload(model)
            del sys.path[0]
            policy['model'] = model.Policy()
            policy['model'].restore_state(best_params)
            policy['model'].to(device)

            for param in policy['model'].parameters():
                print(param.data)
