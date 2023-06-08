import utils
import numpy as np

class rnn:
    def __init__(self, num_input: int, num_hidden: int, num_output: int, alpha):
        self.alpha = alpha
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.truncate = 4
        self.w_hx = np.random.uniform(-np.sqrt(1./self.num_input), np.sqrt(1./num_input), (num_hidden, num_input))
        self.w_hh = np.random.uniform(-np.sqrt(1./self.num_hidden), np.sqrt(1./num_hidden), (num_hidden, num_hidden))
        self.w_oh = np.random.uniform(-np.sqrt(1./self.num_hidden), np.sqrt(1./num_hidden), (num_output, num_hidden))

    def forward(self, X):
        time_steps = len(X)
        self.o_time_steps = np.zeros((time_steps, self.num_output), dtype=float)
        # self.h_time_steps[-1] is t = 0 case ie. the last element
        self.h_time_steps = np.zeros((time_steps+1, self.num_hidden), dtype=float)

        for t in range(time_steps):
            # calculating hidden layer at t
            curr_ht = self.w_hx[:, X[t]] + np.matmul(self.w_hh, self.h_time_steps[t-1])
            curr_ht = np.tanh(curr_ht)
            self.h_time_steps[t] = curr_ht
            
            # calculating output layer at t
            curr_ot = np.matmul(self.w_oh, curr_ht)
            # print(curr_ot)
            curr_ot = utils.softmax(curr_ot)
            # print(curr_ot.shape)
            self.o_time_steps[t] = curr_ot

    def backward(self, X, Y):
        time_steps = len(X)
        self.dLdw_hx = np.zeros(self.w_hx.shape, dtype=float)
        self.dLdw_oh = np.zeros(self.w_oh.shape, dtype=float)
        self.dLdw_hh = np.zeros(self.w_hh.shape, dtype=float)
        # taking base cases to be zero

        dldw_hx = np.zeros(self.w_hx.shape, dtype=float)
        dldw_oh = np.zeros(self.w_oh.shape, dtype=float)
        dldw_hh = np.zeros(self.w_hh.shape, dtype=float)

        # dhprevdw_hx = np.zeros((self.num_hidden*self.num_input, self.num_hidden), dtype=float)
        dhprevdw_hh = np.zeros((self.num_hidden*self.num_hidden, self.num_hidden), dtype=float)

        for t in range(time_steps):
            
            # calculating dldw_oh
            y_hat_y = np.array(self.o_time_steps[t].reshape((self.num_output, 1)))
            y_hat_y[Y[t]] -= 1.0
            dldw_oh = np.matmul(y_hat_y, self.h_time_steps[t].reshape((1, self.num_hidden)))

            # calculating dldw_dhx
            delta = np.matmul(self.w_oh.T, y_hat_y)*(1 - (self.h_time_steps[t].reshape((self.num_hidden, 1)))**2)
            for i in range(t, 0, -1):
                dldw_hx[:, X[i]] += delta.reshape((self.num_hidden, ))
                delta = np.matmul(self.w_hh.T, delta)*(1 - (self.h_time_steps[i-1].reshape((self.num_hidden, 1)))**2)

            # calculating dldw_dhh
            delta = np.matmul(self.w_oh.T, y_hat_y)*(1 - (self.h_time_steps[t].reshape((self.num_hidden, 1)))**2)
            for i in range(t, 0, -1):
                dldw_hh += np.outer(delta, self.h_time_steps[i-1])
                delta = np.matmul(self.w_hh.T, delta)*(1 - (self.h_time_steps[i-1].reshape((self.num_hidden, 1)))**2)
            
            # dhdw_hx = np.matmul(dhprevdw_hx, self.w_hh.T)
            # for i in range(self.num_hidden):
            #     row_start = i*self.num_input
            #     dhdw_hx[row_start + X[t], i] += 1.0
            # dhdw_hx = dhdw_hx*(np.ones(self.num_hidden) - np.power(self.h_time_steps[t], 2))
            # dodw_hx = np.matmul(dhdw_hx, self.w_oh.T)       
            # dldw_hx = np.matmul(dodw_hx, y_hat_y).reshape(self.w_hx.shape)
            # dhprevdw_hx = dhdw_hx

            # dhdw_hh = np.matmul(dhprevdw_hh, self.w_hh.T)
            # for i in range(self.num_hidden):
            #     row_start = i*self.num_hidden
            #     row_end = i*self.num_hidden + self.num_hidden
            #     dhdw_hh[row_start:row_end, i] += self.h_time_steps[t-1]
            # dhdw_hh = dhdw_hh*(np.ones(self.num_hidden) - np.power(self.h_time_steps[t], 2))
            # dodw_hh = np.matmul(dhdw_hh, self.w_oh.T)       
            # dldw_hh = np.matmul(dodw_hh, y_hat_y).reshape(self.w_hh.shape)
            # dhprevdw_hh = dhdw_hh

            self.dLdw_oh += dldw_oh
            self.dLdw_hx += dldw_hx
            self.dLdw_hh += dldw_hh

        self.update_weights()


    def update_weights(self):
        self.w_oh -= (self.alpha*self.dLdw_oh)
        self.w_hx -= (self.alpha*self.dLdw_hx)
        self.w_hh -= (self.alpha*self.dLdw_hh)

    def predict(self):
        return np.argmax(self.o_time_steps[-1])
            
    def loss_t(self):
        pass

    # this fn called after computing one sequence/ series
    def total_loss_of_one_series(self, Y):
        loss = 0.0
        for i in range(self.o_time_steps.shape[0]):
            loss -= np.log(self.o_time_steps[i][Y[i]])
        return loss / len(Y)