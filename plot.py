import matplotlib.pyplot as plt
import pickle
import torch


def plot(rnn_type):
    epoch_losses = [x.cpu() for x in pickle.load(open(f'graph/{rnn_type}/epoch_vs_losses.pkl', 'rb'))][:10]
    # epoch_validation = [x.cpu() for x in pickle.load(open(f'graph/{rnn_type}/epoch_vs_validation.pkl', 'rb'))][:10]

    epochs = [i+1 for i in range(len(epoch_losses))][:10]

    plt.plot(epochs, epoch_losses, label = "training loss")
    # plt.plot(epochs, epoch_validation, label = "validation loss")
    # plt.yticks([x.cpu() for x in torch.arange(0., max(max(epoch_losses), max(epoch_validation)), 0.15)])
    # plt.yticks([x.cpu() for x in torch.arange(0., max(epoch_losses), 0.15)])
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title(rnn_type)
    plt.savefig(f'plots/{rnn_type}.png')
    plt.show()

if __name__ == "__main__":
    plot("mnist")