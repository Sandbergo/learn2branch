import matplotlib.pyplot as plt


def plot_loss(file):
    global_steps = [0]
    train_loss = []
    val_loss = []
    plt.figure(figsize=(20, 8))
    plt.ylim([.0, 5.0])
    plt.plot(global_steps, train_loss, "-", label="Training Loss")
    plt.plot(global_steps, val_loss, "-", label="Validation Loss")
    plt.legend()
    plt.xlabel("Number of gradient steps")
    plt.ylabel("Loss")
    plt.savefig("figures/test.eps")


def plot_kacc():
    pass


if __name__ == "__main__":
    with open('trained_models/cauctions/mlp_sigmoidal_decay/0/log.txt') as file:
        plot_loss(file)
