import vanilla
import lstm
import gru
import torch

if __name__ == '__main__':
    device = "cuda"
    # device = "cpu"

    x = torch.randint(0, 100, (100, )).to(device)
    y = torch.randint(0, 100, (100, )).to(device)
    # x = torch.tensor([0,1]).to(device)
    # y = torch.tensor([1,0]).to(device)

    model = gru.rnn(100, 512, 0.01, device)
    # model = lstm.rnn(2, 2, 0.01, device)
    while True:
        vals = model.forward(x)
        derv = model.backward(x, y, vals)
        model.update_weights(derv)
        loss = model.total_loss_of_one_context(y, vals['o_timesteps'])
        print(loss) 