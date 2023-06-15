import utils, re
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
        self.w_hh1 = np.random.uniform(-np.sqrt(1./self.num_hidden), np.sqrt(1./num_hidden), (num_hidden, num_hidden))
        self.w_hh2 = np.random.uniform(-np.sqrt(1./self.num_hidden), np.sqrt(1./num_hidden), (num_hidden, num_hidden))
        self.w_oh = np.random.uniform(-np.sqrt(1./self.num_hidden), np.sqrt(1./num_hidden), (num_output, num_hidden))

    def forward(self, X):
        time_steps = len(X)
        self.o_time_steps = np.zeros((time_steps, self.num_output), dtype=float)
        # self.h_time_steps[-1] is t = 0 case ie. the last element
        self.h1_time_steps = np.zeros((time_steps+1, self.num_hidden), dtype=float)
        self.h2_time_steps = np.zeros((time_steps+1, self.num_hidden), dtype=float)

        for t in range(time_steps):
            # calculating hidden layer one at t
            curr_h1t = self.w_hx[:, X[t]] + np.matmul(self.w_hh1, self.h1_time_steps[t-1])
            self.h1_time_steps[t] = np.tanh(curr_h1t)

            # calculating hidden layer two at t
            curr_h2t = np.matmul(self.w_hh, self.h1_time_steps[t]) + np.matmul(self.w_hh2, self.h2_time_steps[t-1])
            self.h2_time_steps[t] = np.tanh(curr_h2t)
            
            # calculating output layer at t
            curr_ot = np.matmul(self.w_oh, self.h2_time_steps[t])
            self.o_time_steps[t] = utils.softmax(curr_ot)

    def backward(self, X, Y):
        time_steps = len(X)

        self.dLdw_hx = np.zeros(self.w_hx.shape, dtype=float)
        self.dLdw_oh = np.zeros(self.w_oh.shape, dtype=float)
        self.dLdw_hh = np.zeros(self.w_hh.shape, dtype=float)
        self.dLdw_hh1 = np.zeros(self.w_hh1.shape, dtype=float)
        self.dLdw_hh2 = np.zeros(self.w_hh2.shape, dtype=float)

        for t in range(time_steps-1, -1 , -1):
                
            # calculating dldw_oh
            y_hat_y = np.array(self.o_time_steps[t].reshape((self.num_output, 1)))
            y_hat_y[Y[t]] -= 1.0
            self.dLdw_oh += np.matmul(y_hat_y, self.h2_time_steps[t].reshape((1, self.num_hidden)))

            delta_h2 = np.matmul(self.w_oh.T, y_hat_y)*(1 - (self.h2_time_steps[t].reshape((self.num_hidden, 1)))**2)
            delta_h1 = np.matmul(self.w_hh.T, delta_h2)*(1 - (self.h1_time_steps[t].reshape((self.num_hidden, 1)))**2)
            for i in range(t, -1, -1):
                
                # calculating dldw_hh
                self.dLdw_hh += np.outer(delta_h2, self.h1_time_steps[i])
                
                # calculating dldw_dhx
                self.dLdw_hx[:, X[i]] += delta_h1.reshape((self.num_hidden, ))

                # calculating dldw_hh2
                self.dLdw_hh2 += np.outer(delta_h2, self.h2_time_steps[i-1])

                # calculating dldw_hh1
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

if __name__ == '__main__':
    np.random.seed(2)
    model = rnn(2,2,2, 0.05)
    model.forward([1, 0])
    # print(model.dLdw_hh)
    model.backward([1, 0], [0, 1])
    print(model.dLdw_oh)
    print(model.dLdw_hh)
    print(model.dLdw_hx)
    print(model.dLdw_hh1)
    print(model.dLdw_hh2)
    
    