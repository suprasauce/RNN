from config import *
import mnist_lstm as lstm
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms import Lambda
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pickle

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnn_type = "mnist"

    # prepare training data
    train_dataset = datasets.MNIST(root='mnist_data', train=True,download=True, transform=ToTensor(),
                                   target_transform=Lambda(lambda y: F.one_hot(torch.tensor(y), 10)))
 
    train_dataloader = DataLoader(train_dataset, shuffle=True)

    # initialize model here
    model = lstm.rnn(28, HIDDEN_NEURONS, 10, ALPHA, device)

    epoch_loss = []
    epoch_validation_loss = []
    EPOCHS = 5

    for epoch in range(1, EPOCHS+1):

        iteration = 1
        curr_epoch_loss = 0.

        iteration_obj = iter(train_dataloader)

        while 1:

            try:
                # prepare input
                x, y = next(iteration_obj)
                x = x.to(device)
                y = y.to(device)
                x = x[0][0]

                # run model over input
                vals = model.forward(x)
                derv = model.backward(x, y, vals)
                model.update_weights(derv)
                curr_loss = model.total_loss_of_one_context(y, vals['o_timesteps'])
                
                curr_epoch_loss += (curr_loss / 60000.)

                if iteration%100 == 0:
                    hundred_loss = (curr_epoch_loss*60000.) / iteration
                    print(f"epoch = {epoch}, iteration = {iteration}, loss = {hundred_loss}")

                iteration += 1
            
            except StopIteration:
                break

        print(f"epoch = {epoch}, epoch_loss = {curr_epoch_loss}")

        epoch_loss.append(curr_epoch_loss)

        pickle.dump(model, open(f'models/{rnn_type}/{epoch}_{iteration-1}_{curr_epoch_loss}.pkl', 'wb'))

    pickle.dump(epoch_loss, open(f'graph/{rnn_type}/epoch_vs_losses.pkl', 'wb'))