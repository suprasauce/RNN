import utils
import numpy as np
# import rnn_config as config
# import rnn as r

class lstm_unit:
    def __init__(self, num_hidden, num_input):
        # forget gate
        self.w_fx = np.random.randn(num_hidden, num_input)*0.01
        self.w_fh = np.random.randn(num_hidden, num_hidden)*0.01
        self.b_f = np.zeros((num_hidden, 1))
        self.dLdw_fx = np.zeros(self.w_fx.shape)
        self.dLdw_fh = np.zeros(self.w_fh.shape )
        self.dLdw_bf = np.zeros(self.b_f.shape )

        # input gate
        self.w_ix = np.random.randn(num_hidden, num_input)*0.01
        self.w_ih = np.random.randn(num_hidden, num_hidden)*0.01
        self.b_i = np.zeros((num_hidden, 1))
        self.dLdw_ix = np.zeros(self.w_ix.shape )
        self.dLdw_ih = np.zeros(self.w_ih.shape )
        self.dLdw_bi = np.zeros(self.b_f.shape )

        # ouput gate
        self.w_ox = np.random.randn(num_hidden, num_input)*0.01
        self.w_oh = np.random.randn(num_hidden, num_hidden)*0.01
        self.b_o = np.zeros((num_hidden, 1) )
        self.dLdw_ox = np.zeros(self.w_ox.shape )
        self.dLdw_oh = np.zeros(self.w_oh.shape )
        self.dLdw_bo = np.zeros(self.b_o.shape )

        # input node
        self.w_hx = np.random.randn(num_hidden, num_input)*0.01
        self.w_hh = np.random.randn(num_hidden, num_hidden)*0.01
        self.b_h = np.zeros((num_hidden, 1) )
        self.dLdw_hx = np.zeros(self.w_hx.shape )
        self.dLdw_hh = np.zeros(self.w_hh.shape )
        self.dLdw_bh = np.zeros(self.b_h.shape )

    def get_ct_ht(self, x_t, c_t_1, h_t_1):
        f_t = utils.sigmoid(np.matmul(self.w_fx, x_t) + np.matmul(self.w_fh, h_t_1) + self.b_f) 
        i_t = utils.sigmoid(np.matmul(self.w_ix, x_t) + np.matmul(self.w_ih, h_t_1) + self.b_i)
        o_t = utils.sigmoid(np.matmul(self.w_ox, x_t) + np.matmul(self.w_oh, h_t_1) + self.b_o)
        h_node_t = np.tanh(np.matmul(self.w_hx, x_t) + np.matmul(self.w_hh, h_t_1) + self.b_h)
        c_t = c_t_1*f_t + h_node_t*i_t
        h_t = np.tanh(c_t)*o_t
        return c_t, h_t, f_t, i_t, o_t, h_node_t
    
    def reset_derivatives(self):
        # forget gate
        self.dLdw_fx.fill(0)
        self.dLdw_fh.fill(0)
        self.dLdw_bf.fill(0)
        # input gate
        self.dLdw_ix.fill(0)
        self.dLdw_ih.fill(0)
        self.dLdw_bi.fill(0)
        # ouput gate
        self.dLdw_ox.fill(0)
        self.dLdw_oh.fill(0)
        self.dLdw_bo.fill(0)
        # input node
        self.dLdw_hx.fill(0)
        self.dLdw_hh.fill(0)
        self.dLdw_bh.fill(0)

    def update_weights(self, alpha):
        # forget gate
        self.w_fx -= (alpha*self.dLdw_fx)
        self.w_fh -= (alpha*self.dLdw_fh)
        self.b_f -= (alpha*self.dLdw_bf)
        # input gate
        self.w_ix -= (alpha*self.dLdw_ix)
        self.w_ih -= (alpha*self.dLdw_ih)
        self.b_i -= (alpha*self.dLdw_bi)
        # ouput gate
        self.w_ox -= (alpha*self.dLdw_ox)
        self.w_oh -= (alpha*self.dLdw_oh)
        self.b_o -= (alpha*self.dLdw_bo)
        # input node
        self.w_hx -= (alpha*self.dLdw_hx)
        self.w_hh -= (alpha*self.dLdw_hh)
        self.b_h -= (alpha*self.dLdw_bh)


class rnn:
    def __init__(self, num_input: int, num_hidden: int, num_output: int, alpha):
        self.alpha = alpha
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.lstm_1 = lstm_unit(num_hidden, num_input)
        self.truncate = 10000 # this large val implies, no truncation is being done

        self.w_oh = np.random.randn(self.num_output, self.num_hidden)*0.01
        self.b_o = np.zeros((self.num_output, 1))

    def forward(self, X):
        time_steps = len(X)
        self.o_time_steps = np.zeros((time_steps, self.num_output) )
        # self.h_time_steps[-1] is t = 0 case ie. the last element
        self.h_time_steps = np.zeros((time_steps+1, self.num_hidden) )
        self.c_time_steps = np.zeros((time_steps+1, self.num_hidden) )
        self.h_node_time_steps = np.zeros((time_steps+1, self.num_hidden) )
        self.i_gate_time_steps = np.zeros((time_steps+1, self.num_hidden) )
        self.o_gate_time_steps = np.zeros((time_steps+1, self.num_hidden) )
        self.f_gate_time_steps = np.zeros((time_steps+1, self.num_hidden) )

        for t in range(time_steps):
            # get curr c_t and h_t
            x_t = np.zeros((self.num_input, 1))
            x_t[X[t]] = 1
            c_t, h_t, f_t, i_t, o_gate_t, h_node_t = self.lstm_1.get_ct_ht(x_t, self.c_time_steps[t-1].reshape((self.num_hidden, 1)), self.h_time_steps[t-1].reshape((self.num_hidden, 1)))

            # update list
            self.h_time_steps[t] = h_t.reshape((self.num_hidden, ))
            self.c_time_steps[t] = c_t.reshape((self.num_hidden, ))
            self.h_node_time_steps[t] = h_node_t.reshape((self.num_hidden, ))
            self.i_gate_time_steps[t] = i_t.reshape((self.num_hidden, ))
            self.o_gate_time_steps[t] = o_gate_t.reshape((self.num_hidden, ))
            self.f_gate_time_steps[t] = f_t.reshape((self.num_hidden, ))

            # calculating output layer at t
            curr_ot = np.matmul(self.w_oh, self.h_time_steps[t].reshape((self.num_hidden,1))) + self.b_o
            self.o_time_steps[t] = utils.softmax(curr_ot).reshape((self.num_output, ))
            
    def backward(self, X, Y):
        time_steps = len(X)

        self.dLdw_oh = np.zeros(self.w_oh.shape )
        self.dLdb_o = np.zeros(self.b_o.shape )
        self.lstm_1.reset_derivatives()

        for t in range(time_steps-1, -1 , -1):
                
            # calculating dldw_oh
            y_hat_y = np.array(self.o_time_steps[t].reshape((self.num_output, 1)))
            y_hat_y[Y[t]] -= 1.0
            self.dLdw_oh += np.matmul(y_hat_y, self.h_time_steps[t].reshape((1, self.num_hidden)))
            self.dLdb_o += y_hat_y

            delta_h = np.matmul(self.w_oh.T, y_hat_y).reshape((self.num_hidden, ))
            c_delta = np.zeros((self.num_hidden, ) )

            for i in range(t, max(-1, t - self.truncate - 1), -1):
    
                c_delta += self.o_gate_time_steps[i]*self.f_gate_time_steps[i]*(1-self.c_time_steps[i]**2)*delta_h

                output_gate_delta = np.tanh(self.c_time_steps[i])*self.o_gate_time_steps[i]*(1-self.o_gate_time_steps[i])*delta_h
                forget_gate_delta = c_delta*self.c_time_steps[i-1]*self.f_gate_time_steps[i]*(1-self.f_gate_time_steps[i])
                input_gate_delta = c_delta*self.h_node_time_steps[i]*self.i_gate_time_steps[i]*(1-self.i_gate_time_steps[i])
                input_node_delta = c_delta*self.i_gate_time_steps[i]*(1-self.h_node_time_steps[i]**2)

                # weights connected to current time step input nodes
                self.lstm_1.dLdw_ox[:, X[i]] += output_gate_delta
                self.lstm_1.dLdw_fx[:, X[i]] += forget_gate_delta
                self.lstm_1.dLdw_ix[:, X[i]] += input_gate_delta
                self.lstm_1.dLdw_hx[:, X[i]] += input_node_delta

                # weights connected to previous time step hidden nodes
                self.lstm_1.dLdw_oh += np.outer(output_gate_delta, self.h_time_steps[i-1])
                self.lstm_1.dLdw_fh += np.outer(forget_gate_delta, self.h_time_steps[i-1])
                self.lstm_1.dLdw_ih += np.outer(input_gate_delta, self.h_time_steps[i-1])
                self.lstm_1.dLdw_hh += np.outer(input_node_delta, self.h_time_steps[i-1])

                # biases
                self.lstm_1.dLdw_bo += output_gate_delta.reshape((self.num_hidden, 1))
                self.lstm_1.dLdw_bf += forget_gate_delta.reshape((self.num_hidden, 1))
                self.lstm_1.dLdw_bi += input_gate_delta.reshape((self.num_hidden, 1))
                self.lstm_1.dLdw_bh += input_node_delta.reshape((self.num_hidden, 1))

                delta_h = np.matmul(self.lstm_1.w_oh.T, output_gate_delta) + np.matmul(self.lstm_1.w_fh.T, forget_gate_delta) + \
                            np.matmul(self.lstm_1.w_ih.T, input_gate_delta) + np.matmul(self.lstm_1.w_hh.T, input_node_delta)
                
        self.update_weights()


    def update_weights(self):
        self.w_oh -= (self.alpha*self.dLdw_oh)
        self.b_o -= (self.alpha*self.dLdb_o)
        self.lstm_1.update_weights(self.alpha)
        

    def predict(self):
        pred = self.o_time_steps[-1].ravel()
        index = np.random.choice(range(self.num_output), p = pred)
        return index
            
    def loss_t(self):
        pass

    # this fn called after computing one sequence/ series
    def total_loss_of_one_context(self, Y):
        loss = 0.0
        for i in range(self.o_time_steps.shape[0]):
            loss -= np.log(self.o_time_steps[i][Y[i]])
        return loss / len(Y)
    
# if __name__ == '__main__':
#     # pp = np.array([[3,2,5,6]])
#     # pp = utils.sigmoid(pp)
#     # print(pp)
#     # exit()


#     x = [0,5,6,3,4,6,6]
#     y = [5,6,3,4,6,6,7]
#     lstm = rnn(8, 4, 8, 0.05)
    
#     i = 0
#     while True:
#         lstm.forward(x)
#         # if i == 10:
#         #     print(lstm.lstm_1.dLdw_bf)
#         #     exit()
#         lstm.backward(x,y)
#         # print(lstm.lstm_1.dLdw_bf)
#         print(f"i = {i}, loss = {lstm.total_loss_of_one_context(y)}")
#         i += 1
#         # i = 75k, loss = 0.0001 lstm


