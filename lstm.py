from utils import *
import torch

class lstm_unit:
    def __init__(self, num_hidden, num_input, device):
        self.device = device
        # forget gate
        self.w_fx = (torch.randn((num_hidden, num_input))*0.01).to(self.device)
        self.w_fh = (torch.randn((num_hidden, num_hidden))*0.01).to(self.device)
        self.b_f = torch.zeros((num_hidden, 1)).to(self.device)

        # input gate
        self.w_ix = (torch.randn((num_hidden, num_input))*0.01).to(self.device)
        self.w_ih = (torch.randn((num_hidden, num_hidden))*0.01).to(self.device)
        self.b_i = torch.zeros((num_hidden, 1)).to(self.device)     

        # ouput gate
        self.w_ox = (torch.randn((num_hidden, num_input))*0.01).to(self.device)
        self.w_oh = (torch.randn((num_hidden, num_hidden))*0.01).to(self.device)
        self.b_o = torch.zeros((num_hidden, 1)).to(self.device)

        # input node
        self.w_hx = (torch.randn((num_hidden, num_input))*0.01).to(self.device)
        self.w_hh = (torch.randn((num_hidden, num_hidden))*0.01).to(self.device)
        self.b_h = torch.zeros((num_hidden, 1)).to(self.device)

    def get_ct_ht(self, x_t, c_t_1, h_t_1):
        f_t = torch.sigmoid(torch.matmul(self.w_fx, x_t) + torch.matmul(self.w_fh, h_t_1) + self.b_f) 
        i_t = torch.sigmoid(torch.matmul(self.w_ix, x_t) + torch.matmul(self.w_ih, h_t_1) + self.b_i)
        o_t = torch.sigmoid(torch.matmul(self.w_ox, x_t) + torch.matmul(self.w_oh, h_t_1) + self.b_o)
        h_node_t = torch.tanh(torch.matmul(self.w_hx, x_t) + torch.matmul(self.w_hh, h_t_1) + self.b_h)
        c_t = c_t_1*f_t + h_node_t*i_t
        h_t = torch.tanh(c_t)*o_t
        return c_t, h_t, f_t, i_t, o_t, h_node_t
    
    def update_weights(self, dict: dict, alpha):
        # forget gate
        self.w_fx -= alpha*dict['dw_fx'] 
        self.w_fh -= alpha*dict['dw_fh'] 
        self.b_f -=  alpha*dict['db_f']
        # input gate
        self.w_ix -= alpha*dict['dw_ix'] 
        self.w_ih -= alpha*dict['dw_ih'] 
        self.b_i -=  alpha*dict['db_i']
        # ouput gate
        self.w_ox -= alpha*dict['dw_ox']
        self.w_oh -= alpha*dict['dw_oh']
        self.b_o -=  alpha*dict['db_o']
        # input node
        self.w_hx -= alpha*dict['dw_hx']
        self.w_hh -= alpha*dict['dw_hh']
        self.b_h -=  alpha*dict['db_h']

    def init_derivatives(self):
        dict = {}
        # forget gate
        dict['dw_fx'] = torch.zeros(self.w_fx.shape).to(self.device)
        dict['dw_fh'] = torch.zeros(self.w_fh.shape).to(self.device)
        dict['db_f'] = torch.zeros(self.b_f.shape).to(self.device)
        # input gate
        dict['dw_ix'] = torch.zeros(self.w_ix.shape).to(self.device)
        dict['dw_ih'] = torch.zeros(self.w_ih.shape).to(self.device)
        dict['db_i'] = torch.zeros(self.b_i.shape).to(self.device)
        # ouput gate
        dict['dw_ox'] = torch.zeros(self.w_ox.shape).to(self.device)
        dict['dw_oh'] = torch.zeros(self.w_oh.shape).to(self.device)
        dict['db_o'] = torch.zeros(self.b_o.shape).to(self.device)
        # input node
        dict['dw_hx'] = torch.zeros(self.w_hx.shape).to(self.device)
        dict['dw_hh'] = torch.zeros(self.w_hh.shape).to(self.device)
        dict['db_h'] = torch.zeros(self.b_h.shape).to(self.device)

        return dict

class rnn:
    def __init__(self, num_input: int, num_hidden: int, alpha, device):
        self.device = device
        self.alpha = alpha
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_input
        self.lstm_1 = lstm_unit(num_hidden, num_input, device)
        self.truncate = 10000
        self.w_oh = (torch.randn(self.num_output, self.num_hidden)*0.01).to(self.device)
        self.b_o = torch.zeros((self.num_output, 1)).to(self.device)
        self.temp_out = torch.zeros((self.num_hidden, self.num_hidden)).to(self.device)
        self.clip_value = 1.       

    def forward(self, X):
        timesteps = len(X)
        dict = {}
        
        dict['o_timesteps'] = torch.zeros((timesteps, self.num_output)).to(self.device)
        dict['h1_timesteps'] = torch.zeros((timesteps+1, self.num_hidden)).to(self.device)
        dict['c1_timesteps'] = torch.zeros((timesteps+1, self.num_hidden)).to(self.device)
        dict['h1_node_timesteps'] = torch.zeros((timesteps+1, self.num_hidden)).to(self.device)
        dict['i1_gate_timesteps'] = torch.zeros((timesteps+1, self.num_hidden)).to(self.device)
        dict['o1_gate_timesteps'] = torch.zeros((timesteps+1, self.num_hidden)).to(self.device)
        dict['f1_gate_timesteps'] = torch.zeros((timesteps+1, self.num_hidden)).to(self.device)

        x_t = torch.zeros((self.num_input, 1)).to(self.device)

        for t in range(timesteps):
            x_t[X[t]] = 1.
            c_t, h_t, f_t, i_t, o_gate_t, h_node_t = self.lstm_1.get_ct_ht(x_t, dict['c1_timesteps'][t-1].reshape((self.num_hidden, 1))
                                                                           , dict['h1_timesteps'][t-1].reshape((self.num_hidden, 1)))
              
            # update list
            dict['h1_timesteps'][t] = h_t.reshape((self.num_hidden, ))
            dict['c1_timesteps'][t] = c_t.reshape((self.num_hidden, ))
            dict['h1_node_timesteps'][t] = h_node_t.reshape((self.num_hidden, ))
            dict['i1_gate_timesteps'][t] = i_t.reshape((self.num_hidden, ))
            dict['o1_gate_timesteps'][t] = o_gate_t.reshape((self.num_hidden, ))
            dict['f1_gate_timesteps'][t] = f_t.reshape((self.num_hidden, ))

            # calculating output layer at t
            curr_ot = torch.matmul(self.w_oh, dict['h1_timesteps'][t].reshape((self.num_hidden,1))) + self.b_o
            dict['o_timesteps'][t] = softmax(curr_ot).reshape((self.num_output, ))

            x_t[X[t]] = 0.

        return dict
            
    def backward(self, X, Y, vals: dict):
        timesteps = len(X)
        dict = {}
        
        dict['dlstm'] = self.lstm_1.init_derivatives()
        dict['dw_oh'] = torch.zeros(self.w_oh.shape ).to(self.device)
        dict['db_o'] = torch.zeros(self.b_o.shape ).to(self.device)

        main_delta = torch.zeros((self.num_hidden, )).to(self.device)
        main_c_delta = torch.zeros((self.num_hidden, )).to(self.device)

        for t in range(timesteps-1, -1 , -1):
        
            y_hat_y = vals['o_timesteps'][t].reshape((self.num_output, 1)).clone().detach()
            y_hat_y[Y[t]] -= 1.0
            dict['dw_oh'] += torch.matmul(y_hat_y, vals['h1_timesteps'][t].reshape((1, self.num_hidden)))
            dict['db_o'] += y_hat_y

            delta_h1 = torch.matmul(self.w_oh.T, y_hat_y).reshape((self.num_hidden, ))
            # c_delta1 = torch.zeros((self.num_hidden, ))

            main_delta += delta_h1

            # for i in range(t, max(-1, t - self.truncate - 1), -1):

            # c_delta1 += vals['o1_gate_timesteps'][i]*(1-np.tanh(vals['c1_timesteps'][i])**2)*delta_h1
            main_c_delta += vals['o1_gate_timesteps'][t]*(1-torch.tanh(vals['c1_timesteps'][t])**2)*main_delta

            output_gate_delta1 = main_delta*torch.tanh(vals['c1_timesteps'][t])*vals['o1_gate_timesteps'][t]*(1-vals['o1_gate_timesteps'][t])
            forget_gate_delta1 = main_c_delta*vals['c1_timesteps'][t-1]*vals['f1_gate_timesteps'][t]*(1-vals['f1_gate_timesteps'][t])
            input_gate_delta1 = main_c_delta*vals['h1_node_timesteps'][t]*vals['i1_gate_timesteps'][t]*(1-vals['i1_gate_timesteps'][t])
            input_node_delta1 = main_c_delta*vals['i1_gate_timesteps'][t]*(1-vals['h1_node_timesteps'][t]**2)

            main_delta = torch.matmul(self.lstm_1.w_oh.T, output_gate_delta1) + torch.matmul(self.lstm_1.w_fh.T, forget_gate_delta1) + \
                        torch.matmul(self.lstm_1.w_ih.T, input_gate_delta1) + torch.matmul(self.lstm_1.w_hh.T, input_node_delta1)
            
            # c_delta1 *= vals['f1_gate_timesteps'][i]
            main_c_delta *= vals['f1_gate_timesteps'][t]

            # weights connected to current time step input nodes
            dict['dlstm']['dw_ox'][:, X[t]] += output_gate_delta1
            dict['dlstm']['dw_fx'][:, X[t]] += forget_gate_delta1
            dict['dlstm']['dw_ix'][:, X[t]] += input_gate_delta1
            dict['dlstm']['dw_hx'][:, X[t]] += input_node_delta1

            # weights connected to previous time step hidden nodes

            torch.outer(output_gate_delta1, vals['h1_timesteps'][t-1], out = self.temp_out)
            dict['dlstm']['dw_oh'] += self.temp_out
            torch.outer(forget_gate_delta1, vals['h1_timesteps'][t-1], out = self.temp_out)
            dict['dlstm']['dw_fh'] += self.temp_out
            torch.outer(input_gate_delta1, vals['h1_timesteps'][t-1], out = self.temp_out)
            dict['dlstm']['dw_ih'] +=self.temp_out
            torch.outer(input_node_delta1, vals['h1_timesteps'][t-1], out = self.temp_out)
            dict['dlstm']['dw_hh'] +=self.temp_out

            # biases
            dict['dlstm']['db_o'] += output_gate_delta1.reshape((self.num_hidden, 1))
            dict['dlstm']['db_f'] += forget_gate_delta1.reshape((self.num_hidden, 1))
            dict['dlstm']['db_i'] += input_gate_delta1.reshape((self.num_hidden, 1))
            dict['dlstm']['db_h'] += input_node_delta1.reshape((self.num_hidden, 1))

        return dict
    
    def clip_by_norm(self, derv:dict):
        w_oh = torch.ravel(derv['dw_oh'])
        b_o = torch.ravel(derv['db_o'])
        param = torch.concatenate((w_oh, b_o))

        for i in derv['dlstm'].values():
            temp = i.ravel()
            param = torch.concatenate((param, temp))

        norm = torch.linalg.norm(param)
        if norm <= self.clip_value:
            norm = 1.

        return norm

    def update_weights(self, derv):
        norm = self.clip_by_norm(derv)

        self.w_oh -= self.alpha*(derv['dw_oh'] / norm)
        self.b_o -= self.alpha*(derv['db_o'] / norm)

        for key in derv['dlstm'].keys():
            derv['dlstm'][key] /= norm

        self.lstm_1.update_weights(derv['dlstm'], self.alpha)

    # this fn called after computing one sequence/ series
    def total_loss_of_one_context(self, Y, o_timesteps):
        loss = 0.0
        for i in range(o_timesteps.shape[0]):
            loss -= torch.log(o_timesteps[i][Y[i]])
        return loss / len(Y)