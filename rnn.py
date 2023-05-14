import utils
import tensorflow as tf

class rnn:
    def __init__(self, num_input: int, num_hidden: int, num_output: int, alpha = 0.01):
        self.alpha = alpha
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.w_hx = tf.random.uniform(minval = -tf.sqrt(1./self.num_input), maxval = tf.sqrt(1./self.num_input), shape = (num_hidden, num_input))
        self.w_hh = tf.random.uniform(minval = -tf.sqrt(1./self.num_hidden), maxval = tf.sqrt(1./self.num_hidden), shape = (num_hidden, num_hidden))
        self.w_oh = tf.random.uniform(minval = -tf.sqrt(1./self.num_hidden), maxval = tf.sqrt(1./self.num_hidden), shape = (num_output, num_hidden))

    def forward(self, X):
        time_steps = len(X)
        self.o_time_steps = tf.zeros((time_steps, self.num_output))
        # self.h_time_steps[-1] is t = 0 case ie. the last element
        self.h_time_steps = tf.zeros((time_steps+1, self.num_hidden))

        for t in range(time_steps):
            # calculating hidden layer at t
            curr_ht = tf.gather(params = self.w_hx, indices = X[t], axis=1) + tf.reshape(tf.matmul(self.w_hh, tf.reshape(self.h_time_steps[t-1], (self.num_hidden, 1))), (self.num_hidden,))
            curr_ht = tf.tanh(curr_ht)
            self.h_time_steps = tf.tensor_scatter_nd_update(self.h_time_steps,[[t]], [curr_ht])

            curr_ot = tf.reshape(tf.matmul(self.w_oh, tf.reshape(curr_ht, (self.num_hidden, 1))), (self.num_output, ))
            curr_ot = utils.softmax(curr_ot)
            # self.o_time_steps[t].assign(curr_ot)
            self.o_time_steps = tf.tensor_scatter_nd_update(self.o_time_steps,[[t]], [curr_ot])

    def backward(self, X, Y):
        time_steps = len(X)
        self.dLdw_hx = tf.zeros(self.w_hx.shape)
        self.dLdw_oh = tf.zeros(self.w_oh.shape)
        self.dLdw_hh = tf.zeros(self.w_hh.shape)
        # taking base cases to be zero

        dhprevdw_hx = tf.zeros((self.num_hidden*self.num_input, self.num_hidden))
        dhprevdw_hh = tf.zeros((self.num_hidden*self.num_hidden, self.num_hidden))
    
        for t in range(time_steps):
            y_hat_y = tf.reshape(self.o_time_steps[t], (self.num_output, 1))
            # y_hat_y = y_hat_y[Y[t]].assign(y_hat_y[Y[t]] - 1.0)
            y_hat_y = tf.tensor_scatter_nd_update(y_hat_y,[[Y[t]]], [y_hat_y[Y[t]] - 1.0])
            dldw_oh = tf.matmul(y_hat_y, tf.reshape(self.h_time_steps[t], (1, self.num_hidden)))    
            
            # from here
            dhdw_hx = tf.matmul(dhprevdw_hx, tf.transpose(self.w_hh))
            for i in range(self.num_hidden):
                row_start = i*self.num_input
                index = [row_start + X[t], i]
                dhdw_hx = tf.tensor_scatter_nd_update(dhdw_hx, [index], [dhdw_hx[index] + 1.0])
            dhdw_hx = tf.multiply(dhdw_hx, tf.subtract(tf.ones(self.num_hidden), tf.pow(self.h_time_steps[t], 2)))
            dodw_hx = tf.matmul(dhdw_hx, tf.transpose(self.w_oh))       
            dldw_hx = tf.matmul(dodw_hx, y_hat_y)
            dldw_hx = tf.reshape(dldw_hx, self.w_hx.shape)
            dhprevdw_hx = dhdw_hx

            dhdw_hh = tf.Variable(tf.matmul(dhprevdw_hh, tf.transpose(self.w_hh)))
            for i in range(self.num_hidden):
                row_start = i*self.num_hidden
                row_end = i*self.num_hidden + self.num_hidden
                dhdw_hh = dhdw_hh[row_start:row_end, i].assign(self.h_time_steps[t])
            dhdw_hh = tf.multiply(dhdw_hh, tf.subtract(tf.ones(self.num_hidden), tf.pow(self.h_time_steps[t], 2)))
            dodw_hh = tf.matmul(dhdw_hh, tf.transpose(self.w_oh))       
            dldw_hh = tf.matmul(dodw_hh, y_hat_y)
            dldw_hh = tf.reshape(dldw_hh, self.w_hh.shape)
            dhprevdw_hh = dhdw_hh

            self.dLdw_oh = tf.add(self.dLdw_oh, dldw_oh)
            self.dLdw_hx = tf.add(self.dLdw_hx, dldw_hx)
            self.dLdw_hh = tf.add(self.dLdw_hh, dldw_hh)

        self.update_weights()

    def update_weights(self):
        self.w_oh = tf.subtract(self.w_oh, 1.0*self.alpha*self.dLdw_oh)
        self.w_hx = tf.subtract(self.w_hx, 1.0*self.alpha*self.dLdw_hx)
        self.w_hh = tf.subtract(self.w_hh, 1.0*self.alpha*self.dLdw_hh)
            
    def loss_t(self):
        pass

    # this fn called after computing one sequence/ series
    def total_loss_of_one_series(self, Y):
        loss = 0.0
        for i in range(self.o_time_steps.shape[0]):
            loss -= tf.math.log(self.o_time_steps[i][Y[i]])
        return loss / len(Y)