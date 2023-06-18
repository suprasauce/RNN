import utils
import numpy as np


class rnn:
    def __init__(self, num_input: int, num_hidden: int, num_output: int, alpha):
        self.alpha = alpha
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.truncate = 10000 # this large val implies, no truncation is being done
        self.w_hx = np.random.randn(self.num_hidden, self.num_input)*0.01
        self.w_hh = np.random.randn(self.num_hidden, self.num_hidden)*0.01
        self.w_hh1 = np.random.randn(self.num_hidden, self.num_hidden)*0.01
        self.w_hh2 = np.random.randn(self.num_hidden, self.num_hidden)*0.01
        self.w_oh = np.random.randn(self.num_output, self.num_hidden)*0.01
        self.b_h1 = np.zeros((self.num_hidden, 1))
        self.b_h2 = np.zeros((self.num_hidden, 1))
        self.b_o = np.zeros((self.num_output, 1))

    def forward(self, X):
        time_steps = len(X)
        self.o_time_steps = np.zeros((time_steps, self.num_output), dtype=float)
        # self.h_time_steps[-1] is t = 0 case ie. the last element
        self.h1_time_steps = np.zeros((time_steps+1, self.num_hidden), dtype=float)
        self.h2_time_steps = np.zeros((time_steps+1, self.num_hidden), dtype=float)

        for t in range(time_steps):
            # calculating hidden layer one at t
            curr_h1t = self.w_hx[:, X[t]] + np.matmul(self.w_hh1, self.h1_time_steps[t-1])
            curr_h1t += self.b_h1.reshape((self.num_hidden, ))
            self.h1_time_steps[t] = np.tanh(curr_h1t)

            # calculating hidden layer two at t
            curr_h2t = np.matmul(self.w_hh, self.h1_time_steps[t]) + np.matmul(self.w_hh2, self.h2_time_steps[t-1])
            curr_h2t += self.b_h2.reshape((self.num_hidden, ))
            self.h2_time_steps[t] = np.tanh(curr_h2t)
            
            # calculating output layer at t
            curr_ot = np.matmul(self.w_oh, self.h2_time_steps[t])
            curr_ot += self.b_o.reshape((self.num_output, ))
            self.o_time_steps[t] = utils.softmax(curr_ot)

    def backward(self, X, Y):
        time_steps = len(X)

        self.dLdw_hx = np.zeros(self.w_hx.shape, dtype=float)
        self.dLdw_oh = np.zeros(self.w_oh.shape, dtype=float)
        self.dLdw_hh = np.zeros(self.w_hh.shape, dtype=float)
        self.dLdw_hh1 = np.zeros(self.w_hh1.shape, dtype=float)
        self.dLdw_hh2 = np.zeros(self.w_hh2.shape, dtype=float)
        self.dLdb_o = np.zeros(self.b_o.shape, dtype=float)
        self.dLb_h2 = np.zeros(self.b_h2.shape, dtype=float)
        self.dLdb_h1 = np.zeros(self.b_h1.shape, dtype=float)

        for t in range(time_steps-1, -1 , -1):
                
            # calculating dldw_oh
            y_hat_y = np.array(self.o_time_steps[t].reshape((self.num_output, 1)))
            y_hat_y[Y[t]] -= 1.0
            self.dLdw_oh += np.matmul(y_hat_y, self.h2_time_steps[t].reshape((1, self.num_hidden)))
            self.dLdb_o += y_hat_y

            delta_h2 = np.matmul(self.w_oh.T, y_hat_y)*(1 - (self.h2_time_steps[t].reshape((self.num_hidden, 1)))**2)
            delta_h1 = np.matmul(self.w_hh.T, delta_h2)*(1 - (self.h1_time_steps[t].reshape((self.num_hidden, 1)))**2)

            for i in range(t, max(-1, t - self.truncate - 1), -1):

                self.dLb_h2 += delta_h2
                self.dLdb_h1 += delta_h1                
                self.dLdw_hh += np.outer(delta_h2, self.h1_time_steps[i])
                self.dLdw_hx[:, X[i]] += delta_h1.reshape((self.num_hidden, ))
                self.dLdw_hh2 += np.outer(delta_h2, self.h2_time_steps[i-1])
                self.dLdw_hh1 += np.outer(delta_h1, self.h1_time_steps[i-1])
                
                delta_h2 = np.matmul(self.w_hh2.T, delta_h2)*(1 - (self.h2_time_steps[i-1].reshape((self.num_hidden, 1)))**2)
                delta_h1 = np.matmul(self.w_hh1.T, delta_h1)*(1 - (self.h1_time_steps[i-1].reshape((self.num_hidden, 1)))**2)

        self.update_weights()


    def update_weights(self):
        self.w_oh -= (self.alpha*self.dLdw_oh)
        self.w_hx -= (self.alpha*self.dLdw_hx)
        self.w_hh -= (self.alpha*self.dLdw_hh)
        self.w_hh1 -= (self.alpha*self.dLdw_hh1)
        self.w_hh2 -= (self.alpha*self.dLdw_hh2)
        self.b_o -= (self.alpha*self.dLdb_o)
        self.b_h1 -= (self.alpha*self.dLdb_h1)
        self.b_h2 -= (self.alpha*self.dLb_h2)
        

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