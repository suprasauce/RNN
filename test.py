# import vanilla
import mnist_lstm as lstm
# import gru
import torch

if __name__ == '__main__':
    # device = "cuda"
    device = "cpu"

    # x = torch.rand((28, 28)).to(device)
    # y = torch.zeros((10, )).to(device)
    # y[2] = 1.
    x = torch.tensor([[0.1,0.2], [0.8,0.9]], dtype=torch.float).to(device)
    y = torch.tensor([0,1], dtype=torch.float).to(device)
    
    # model = lstm.rnn(28, 512, 10, 0.01, device)
    model = lstm.rnn(2, 2, 2, 0.1, device)
    while True:
        vals = model.forward(x)
        derv = model.backward(x, y, vals)
        model.update_weights(derv)
        # print(vals['o_timesteps'])
        loss = model.total_loss_of_one_context(y, vals['o_timesteps'])
        print(loss) 