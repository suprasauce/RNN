import torch
from utils import *

class gru:
    def __init__(self, num_input, num_hidden, device):
        self.device = device
        # reset gate
        self.num_hidden = num_hidden
        self.w_rx = (torch.randn((num_hidden, num_input))*0.01).to(device)
        self.w_rh = (torch.randn((num_hidden, num_hidden))*0.01).to(device)
        self.b_r = torch.zeros((num_hidden, 1)).to(device)

        # update gate
        self.w_ux = (torch.randn((num_hidden, num_input))*0.01).to(device)
        self.w_uh = (torch.randn((num_hidden, num_hidden))*0.01).to(device)
        self.b_u = torch.zeros((num_hidden, 1)).to(device)

        # input node
        self.w_ix = (torch.randn((num_hidden, num_input))*0.01).to(device)
        self.w_ih = (torch.randn((num_hidden, num_hidden))*0.01).to(device)
        self.b_i = torch.zeros((num_hidden, 1)).to(device)

    def get_h_t(self, x_t, h_t_1):
        r_t = torch.sigmoid(torch.matmul(self.w_rx, x_t) + torch.matmul(self.w_rh, h_t_1) + self.b_r)
        u_t = torch.sigmoid(torch.matmul(self.w_ux, x_t) + torch.matmul(self.w_uh, h_t_1) + self.b_u)
        i_t = torch.tanh(torch.matmul(self.w_ix, x_t) + torch.matmul(self.w_ih, (h_t_1*r_t)) + self.b_i)
        h_t = torch.mul(u_t, h_t_1) + torch.mul((1. - u_t), i_t)
        return h_t.reshape(self.num_hidden, ), i_t.reshape(self.num_hidden, ), u_t.reshape(self.num_hidden, ), r_t.reshape(self.num_hidden, )

    def init_derv(self):
        dict = {}
        dict['dw_rx'] = torch.zeros(self.w_rx.shape).to(self.device)
        dict['dw_ux'] = torch.zeros(self.w_ux.shape).to(self.device)
        dict['dw_ix'] = torch.zeros(self.w_ix.shape).to(self.device)

        dict['dw_rh'] = torch.zeros(self.w_rh.shape).to(self.device)
        dict['dw_uh'] = torch.zeros(self.w_uh.shape).to(self.device)
        dict['dw_ih'] = torch.zeros(self.w_ih.shape).to(self.device)

        dict['db_r'] = torch.zeros(self.b_r.shape).to(self.device)
        dict['db_u'] = torch.zeros(self.b_u.shape).to(self.device)
        dict['db_i'] = torch.zeros(self.b_i.shape).to(self.device)

        return dict

    def update_weights(self, dict: dict, alpha):
        # reset gate
        self.w_rx -= alpha*dict['dw_rx']
        self.w_rh -= alpha*dict['dw_rh']
        self.b_r  -= alpha*dict['db_r']

        # update gate
        self.w_ux -= alpha*dict['dw_ux']
        self.w_uh -= alpha*dict['dw_uh']
        self.b_u  -= alpha*dict['db_u']

        # input node
        self.w_ix -= alpha*dict['dw_ix']
        self.w_ih -= alpha*dict['dw_ih']
        self.b_i  -= alpha*dict['db_i']

class rnn:
    def __init__(self, num_input, num_hidden, alpha, device):
        self.device = device
        self.alpha = alpha
        self.truncate = 10000
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.gru = gru(num_input, num_hidden, device)
        self.w_oh = (torch.randn((num_input, num_hidden))*0.01).to(device)
        self.b_o = torch.zeros((num_input, 1)).to(device)
        self.temp_out = torch.zeros((self.num_hidden, self.num_hidden)).to(device)
        self.clip_value = 1.

    def forward(self, X):
        time_steps = len(X)
        dict = {}
        dict['h_timesteps'] = torch.zeros((time_steps+1, self.num_hidden)).to(self.device)
        dict['o_timesteps'] = torch.zeros((time_steps, self.num_input)).to(self.device)
        dict['i_timesteps'] = torch.zeros((time_steps, self.num_hidden)).to(self.device)
        dict['u_timesteps'] = torch.zeros((time_steps, self.num_hidden)).to(self.device)
        dict['r_timesteps'] = torch.zeros((time_steps, self.num_hidden)).to(self.device)
        x_t = torch.zeros((self.num_input, 1)).to(self.device)

        for i in range(time_steps):
            x_t[X[i]] = 1.
            dict['h_timesteps'][i], dict['i_timesteps'][i], dict['u_timesteps'][i], dict['r_timesteps'][i] \
                = self.gru.get_h_t(x_t, dict['h_timesteps'][i-1].reshape(self.num_hidden, 1).clone().detach())
            dict['o_timesteps'][i] = softmax(torch.matmul(self.w_oh, dict['h_timesteps'][i].reshape((self.num_hidden,1))) + self.b_o).reshape((self.num_input, ))
            x_t[X[i]] = 0.

        return dict

    def backward(self, X, Y, dict: dict):
        time_steps = len(X)
        derv = {}
        derv['dw_oh'] = torch.zeros(self.w_oh.shape).to(self.device)
        derv['db_o'] = torch.zeros(self.b_o.shape).to(self.device)
        derv['dgru'] = self.gru.init_derv()

        main_delta = torch.zeros((self.num_hidden, )).to(self.device)

        for t in range(time_steps-1, -1 , -1):
            y_hat_y = dict['o_timesteps'][t].reshape((self.num_input, 1)).clone().detach()
            y_hat_y[Y[t]] -= 1.0
            derv['dw_oh'] += torch.matmul(y_hat_y, dict['h_timesteps'][t].reshape((1, self.num_hidden)))
            derv['db_o'] += y_hat_y

            delta = torch.matmul(self.w_oh.T, y_hat_y).reshape((self.num_hidden, ))

            main_delta += delta

            update_gate_delta = (dict['h_timesteps'][t-1] - dict['i_timesteps'][t])*dict['u_timesteps'][t]*(1 - dict['u_timesteps'][t])*main_delta
            input_gate_delta = (1 - dict['u_timesteps'][t])*(1. - dict['i_timesteps'][t]**2)*main_delta
            reset_gate_delta = torch.matmul(self.gru.w_ih, input_gate_delta)*dict['r_timesteps'][t]*(1. - dict['r_timesteps'][t])*dict['h_timesteps'][t-1]

            main_delta = torch.matmul(self.gru.w_uh, update_gate_delta) + torch.matmul(self.gru.w_ih, input_gate_delta)*dict['r_timesteps'][t] + \
            torch.matmul(self.gru.w_rh, reset_gate_delta) + dict['u_timesteps'][t]*main_delta

            derv['dgru']['dw_rx'][:, X[t].type(torch.int)] += reset_gate_delta
            derv['dgru']['dw_ux'][:, X[t].type(torch.int)] += update_gate_delta
            derv['dgru']['dw_ix'][:, X[t].type(torch.int)] += input_gate_delta

            torch.outer(reset_gate_delta, dict['h_timesteps'][t-1], out = self.temp_out)
            derv['dgru']['dw_rh'] += self.temp_out
            torch.outer(update_gate_delta, dict['h_timesteps'][t-1], out = self.temp_out)
            derv['dgru']['dw_uh'] += self.temp_out
            torch.outer(input_gate_delta, (dict['h_timesteps'][t-1]*dict['r_timesteps'][t]), out = self.temp_out)
            derv['dgru']['dw_ih'] += self.temp_out

            derv['dgru']['db_r'] += reset_gate_delta.reshape(self.num_hidden, 1)
            derv['dgru']['db_u'] += update_gate_delta.reshape(self.num_hidden, 1)
            derv['dgru']['db_i'] += input_gate_delta.reshape(self.num_hidden, 1)

        return derv

    def clip_by_norm(self, derv:dict):
        w_oh = torch.ravel(derv['dw_oh'])
        b_o = torch.ravel(derv['db_o'])
        param = torch.concatenate((w_oh, b_o))
    
        for i in derv['dgru'].values():
            temp = i.ravel()
            param = torch.concatenate((param, temp))

        norm = torch.linalg.norm(param)
        if norm <= self.clip_value:
            norm = 1.

        return norm

    def update_weights(self, derv):
        # perform gradient clipping
        norm = self.clip_by_norm(derv)

        self.w_oh -= self.alpha*(derv['dw_oh']/ norm)
        self.b_o -= self.alpha*(derv['db_o']/ norm)

        for key in derv['dgru'].keys():
            derv['dgru'][key] /= norm
        self.gru.update_weights(derv['dgru'], self.alpha)

    def total_loss_of_one_context(self, Y, o_timesteps):
        loss = 0.
        for i in range(o_timesteps.shape[0]):
            loss -= torch.log(o_timesteps[i][Y[i]])
        return loss / len(Y)


