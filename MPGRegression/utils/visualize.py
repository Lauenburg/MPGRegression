import matplotlib.pyplot as plt


def plot_loss(history, y_lim_bot: int = 0, y_lim_top: int = 100):
    """Plot"""
    plt.plot(history["loss"], label="loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.ylim([y_lim_bot, y_lim_top])
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.show()
