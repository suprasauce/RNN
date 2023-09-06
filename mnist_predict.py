from config import *
import mnist_lstm as lstm
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms import Lambda
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pickle

def predict(out):
    return torch.argmax(out)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnn_type = "mnist"

    # prepare training data
    train_dataset = datasets.MNIST(root='mnist_data', train=False,download=True, transform=ToTensor())

    train_dataloader = DataLoader(train_dataset, shuffle=True)

    # initialize model here
    # model = lstm.rnn(28, HIDDEN_NEURONS, 10, ALPHA, device)
    model = pickle.load(open('models/mnist/4_60000_0.15363748371601105.pkl', 'rb'))

    iteration_obj = iter(train_dataloader)
    correct = 0
    iteration = 1

    while 1:

        try:
            # prepare input
            x, y = next(iteration_obj)
            x = x.to(device)
            y = y.to(device)
            x = x[0][0]

            # run model over input
            vals = model.forward(x)
            pred = predict(vals['o_timesteps'][-1])
            print(iteration, pred, y)
            if pred == y:
                correct += 1
            iteration += 1
        
        except StopIteration:
            break
    print(correct)