import matplotlib.pyplot as plt
import re


def plot_loss(str_log):

    train_regex = re.compile('TRAIN LOSS: [0-9.]{5}')
    train_loss = train_regex.findall(str_log)
    train_loss = [float(x[-5:]) for x in train_loss]

    val_regex = re.compile('VALID LOSS: [0-9.]{5}')
    val_loss = val_regex.findall(str_log)
    val_loss = [float(x[-5:]) for x in val_loss]

    global_steps_train = list(range(0, len(train_loss)*312, 312))
    global_steps_val = list(range(0, len(val_loss)*312, 312))

    plt.figure(figsize=(8, 4))
    plt.ylim([2.0, 4.0])
    plt.plot(global_steps_train, train_loss, "-", label="Training Loss")
    plt.plot(global_steps_val, val_loss, "-", label="Validation Loss")
    plt.legend()
    plt.xlabel("Number of gradient steps")
    plt.ylabel("Loss")
    plt.savefig("figures/loss0.eps")


def plot_kacc(str_log):
    regex = re.compile('@1: [0-9.]{5}')
    acc = regex.findall(str_log)
    acc = [float(x[-5:]) for x in acc]

    global_steps = list(range(0, len(acc)*312, 312))

    plt.figure(figsize=(8, 4))
    plt.ylim([0.0, 0.5])
    plt.plot(global_steps, acc, "-", label="acc@1")
    plt.legend()
    plt.xlabel("Number of gradient steps")
    plt.ylabel("acc@1")
    plt.savefig("figures/kacc_graph.png")


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
    with open('trained_models/cauctions/mlp_sigmoidal_decay/0/log.txt') as log:
        str_log = log.read()
        plot_loss(str_log)
        #plot_kacc(str_log)
        #
        """with open('trained_models/cauctions/mlp_sigmoidal_decay/15/log.txt') as rand:
            str_random = rand.read()
            barplot_kacc_random(str_log, str_random)"""
