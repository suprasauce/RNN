from utils import *
import torch

class rnn:
    def __init__(self, num_input: int, num_hidden: int, alpha, device):
        self.device = device
        self.alpha = alpha
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_input
        self.truncate = 10000
        self.w_hx = (torch.randn((self.num_hidden, self.num_input))*0.01).to(self.device)
        self.w_hh1 = (torch.randn((self.num_hidden, self.num_hidden))*0.01).to(self.device)
        self.w_oh = (torch.randn((self.num_output, self.num_hidden))*0.01).to(self.device)
        self.b_h1 = torch.zeros((self.num_hidden, 1)).to(self.device)
        self.b_o = torch.zeros((self.num_output, 1)).to(self.device)
        self.clip_value = 1.

    def forward(self, X):
        time_steps = len(X)
        dict = {}
        dict['o_timesteps'] = torch.zeros((time_steps, self.num_output)).to(self.device)
        dict['h1_timesteps'] = torch.zeros((time_steps+1, self.num_hidden)).to(self.device)

        for t in range(time_steps):
            # calculating hidden layer one at t
            curr_h1t = self.w_hx[:, X[t]] + torch.matmul(self.w_hh1, dict['h1_timesteps'][t-1])
            curr_h1t += self.b_h1.reshape((self.num_hidden, ))
            dict['h1_timesteps'][t] = torch.tanh(curr_h1t)
            
            # calculating output layer at t
            curr_ot = torch.matmul(self.w_oh, dict['h1_timesteps'][t])
            curr_ot += self.b_o.reshape((self.num_output, ))
            dict['o_timesteps'][t] = softmax(curr_ot)

        return dict

    def backward(self, X, Y, vals: dict):
        time_steps = len(X)
        dict = {}

        dict['dLdw_hx'] = torch.zeros(self.w_hx.shape).to(self.device)
        dict['dLdw_oh'] = torch.zeros(self.w_oh.shape).to(self.device)
        dict['dLdw_hh1'] = torch.zeros(self.w_hh1.shape).to(self.device)
        dict['dLdb_o'] = torch.zeros(self.b_o.shape).to(self.device)
        dict['dLdb_h1'] = torch.zeros(self.b_h1.shape).to(self.device)

        main_delta = torch.zeros((self.num_hidden, 1)).to(self.device)

        for t in range(time_steps-1, -1 , -1):
                
            # calculating dldw_oh
            y_hat_y = vals['o_timesteps'][t].reshape((self.num_output, 1)).clone().detach()
            y_hat_y[Y[t]] -= 1.0
            dict['dLdw_oh'] += torch.matmul(y_hat_y, vals['h1_timesteps'][t].reshape((1, self.num_hidden)))
            dict['dLdb_o'] += y_hat_y

            delta_h1 = torch.matmul(self.w_oh.T, y_hat_y)*(1 - (vals['h1_timesteps'][t].reshape((self.num_hidden, 1)))**2)
            main_delta += delta_h1

            dict['dLdb_h1'] += main_delta            
            dict['dLdw_hx'][:, X[t]] += main_delta.reshape((self.num_hidden, ))
            dict['dLdw_hh1'] += torch.outer(main_delta.reshape(self.num_hidden, ), vals['h1_timesteps'][t-1])
            
            main_delta = torch.matmul(self.w_hh1.T, main_delta)*(1 - (vals['h1_timesteps'][t-1].reshape((self.num_hidden, 1)))**2)
        
        return dict
    
    def clip_by_norm(self, derv:dict):

        param = torch.tensor([]).to(self.device)
        for i in derv.values():
            temp = i.ravel()
            param = torch.concatenate((param, temp))

        norm = torch.linalg.norm(param)
        if norm <= self.clip_value:
            norm = 1.

        return norm

    def update_weights(self, derv):
        norm = self.clip_by_norm(derv)

        self.w_oh -= (self.alpha*(derv['dLdw_oh'] / norm))
        self.w_hx -= (self.alpha*(derv['dLdw_hx'] / norm))
        self.w_hh1 -= (self.alpha*(derv['dLdw_hh1'] / norm))
        self.b_o -= (self.alpha*(derv['dLdb_o'] / norm))
        self.b_h1 -= (self.alpha*(derv['dLdb_h1'] / norm))

    def total_loss_of_one_context(self, Y, o_timesteps):
        loss = 0.0
        for i in range(o_timesteps.shape[0]):
            loss -= torch.log(o_timesteps[i][Y[i]])
        return loss / len(Y)