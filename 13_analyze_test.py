import matplotlib.pyplot as plt
import re


def barplot_kacc(str_log):
    regex = re.compile('@[0-9]{1,2}: [0-9.]{5}')
    acc = regex.findall(str_log)
    acc = [float(x[-5:]) for x in acc]

    import seaborn as sns
    import pandas as pd
    sns.set_theme(style="whitegrid")
    acc_df = pd.DataFrame(
        {'acc@1': [acc[-4]],
         'acc@3': [acc[-3]],
         'acc@5': [acc[-2]],
         'acc@10': [acc[-1]]}
        )
    ax = sns.barplot(data=acc_df, ci=None)
    ax.figure.savefig("figures/barplot.png")


def barplot_kacc_random(str_log):
    regex = re.compile('@[0-9]{1,2}: [0-9.]{5}')
    acc = regex.findall(str_log)
    acc = [float(x[-5:]) for x in acc]

    import seaborn as sns
    import pandas as pd
    sns.set_theme(style="whitegrid")
    acc_df = pd.DataFrame(
        {'acc@1': [acc[-4]],
         'acc@3': [acc[-3]],
         'acc@5': [acc[-2]],
         'acc@10': [acc[-1]]}
        )
    print(acc_df)
    ax = sns.barplot(data=acc_df, ci=None)
    ax.figure.savefig("figures/barplot.png")


if __name__ == "__main__":
    with open('trained_models/cauctions/mlp_sigmoidal_decay/12/log.txt') as log:
        str_log = log.read()

        with open('trained_models/cauctions/mlp_sigmoidal_decay/15/log.txt') as rand:
            str_random = rand.read()
            barplot_kacc_random(str_log, str_random)
