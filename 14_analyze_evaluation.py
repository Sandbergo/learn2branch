"""
Create data for tables based on test and evaluation results.
"""

import os
import pandas as pd


def console_print():
    results = os.listdir('eval_results/cauctions/')
    results = [pd.read_csv(f'eval_results/cauctions/{file}') for file in results]

    types = ['small', 'medium', 'big']

    for result in results:
        print(f"\npolicy: {result['policy'][0]} \t\t problem: {result['problem'][0]}\t\t seed: {result['problem'][0]}")
        for type in types:
            times = result.loc[result['type'] == type]['walltime']
            nnodes = result.loc[result['type'] == type]['nnodes']
            status = result.loc[result['type'] == type].groupby(result.status.str.strip("'"))['status'].count()
            print(f'\t{type}: {round(times.mean(), 2)} +/- {round(times.std()/times.mean(), 2)*100}% s', end='  ')
            print(f'\tnodes: {round(nnodes.mean(), 1)} +/- {round(nnodes.std()/times.mean(), 1)*100}%  #', end='  ')
            try:
                print(f'''\topt: {status['optimal']} /  {status['timelimit']}''')
            except Exception:
                pass
            print()


def barplot():
    results = os.listdir('eval_results/cauctions/')
    results = [pd.read_csv(f'eval_results/cauctions/{file}') for file in results]

    types = ['small', 'medium', 'big']


if __name__ == '__main__':
    console_print()
    # barplot()
