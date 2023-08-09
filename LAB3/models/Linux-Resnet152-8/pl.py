import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("learning_curve.csv").to_numpy()


all_activation_history = [[df[:, 2].tolist(), df[:, 3], df[:, 4], df[:, 5]]]


def plot_acc_loss(all_activation_history, epochs, save_path):
    colors = [['red', 'orange'], ['blue', 'cyan'], ['green', 'purple']]
    labels = [['resnet18_train', 'resnet18_valid'], [
        'leaky_relu_train', 'leaky_relu_test'], ['elu_train', 'elu_test']]
    fig, axs = plt.subplots(2, 1)
    for color_idx, (training_acc_history, valid_acc_history, training_loss_history, valid_loss_history) in enumerate(all_activation_history):
        axs[0].plot(range(1, epochs + 1), training_acc_history, color=colors[color_idx]
                    [0], label=labels[color_idx][0] + '_acc', linestyle='-')
        axs[0].plot(range(1, epochs + 1), valid_acc_history, color=colors[color_idx]
                    [1], label=labels[color_idx][1] + '_acc', linestyle='-')

        axs[1].plot(range(1, epochs + 1), training_loss_history, color=colors[color_idx]
                    [0], label=labels[color_idx][0] + '_loss', linestyle='-')
        axs[1].plot(range(1, epochs + 1), valid_loss_history, color=colors[color_idx]
                    [1], label=labels[color_idx][1] + '_loss', linestyle='-')
    axs[0].set_title('Accuracy curves')
    axs[1].set_title('Loss curves')
    axs[0].set_ylabel('Accuracy')
    axs[1].set_ylabel('Loss')
    axs[0].legend(loc='upper left')
    axs[1].legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f"learning_curve.jpg")
    plt.show()


plot_acc_loss(all_activation_history, 10, ".")
