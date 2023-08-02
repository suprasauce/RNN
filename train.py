import rnn_config as config
import numpy as np
import pickle
from lstm import rnn

class train_class:
    def __init__(self, load) -> None:
        self.dataset = self.get_dataset()
        self.vocab = sorted(list(set(self.dataset)))
        self.model = rnn(len(self.vocab), config.HIDDEN_NEURONS, len(self.vocab), config.ALPHA) if load else self.load_model()

    def load_model(self):
        return pickle.load(open('models/model_loss_0.6545.pkl', 'rb'))

    def get_dataset(self):
        dataset = ''
        with open('dataset/dataset.txt', 'r') as f:
            dataset = f.read()
        return dataset

    def sample(self, n, seq): # n = length of predicted seq including i, i = initial seq
        x = [self.vocab.index(c) for c in seq]
        while n:
            self.model.forward(x[-min(len(x), config.CONTEXT_LENGTH):])
            x.append(self.model.predict())
            n -= 1
        x = [self.vocab[i] for i in x]
        x = ''.join(x)
        print(x)

    def run(self, epochs):
        for i in range(epochs):
            '''
            rather than sliding window approach, giving a random offset when it shifts from
            one example to another, for ex: for one example, start index is 'n' then for next example
            start index will be 'n + random(1, k)'
            '''
            curr_index = 0
            iter = 1
            while True:
                x, y, curr_index, run = self.get_example(curr_index, iter)
                # print(x,y)
                if run == False:
                    break
                self.model.forward(x)
                self.model.backward(x, y)
                if iter%100 == 0:
                    loss = self.model.total_loss_of_one_context(y)
                    print('-------------------------------')
                    print(f'epoch = {i}, example = {iter}, loss = {loss}')
                    self.sample(50, 'I ')
                    print('-------------------------------')
                    self.save_model(loss)
                iter += 1

    def get_example(self, curr_index, iter):
        if iter == 1:
            s = curr_index
            e = s + config.CONTEXT_LENGTH
            if e > len(self.dataset)-1:
                return [], [], 0, False
            return self.tokenize(self.dataset[s:e]), self.tokenize(self.dataset[s+1:e+1]), s, True
        else:
            s = curr_index + np.random.randint(config.CONTEXT_LENGTH/10, config.CONTEXT_LENGTH/5 + 1)
            e = s + config.CONTEXT_LENGTH
            if e > len(self.dataset)-1:
                return [], [], 0, False
            return self.tokenize(self.dataset[s:e]), self.tokenize(self.dataset[s+1:e+1]), s, True
            
    def tokenize(self, v):
        return [self.vocab.index(c) for c in v]
    
    def save_model(self, loss):
        pickle.dump(self.model, open(f'models/model_loss_{round(loss, 4)}.pkl', 'wb'))

if __name__ == '__main__':
    train = train_class(True)
    train.run(100)
    # train.sample(1000,'stat')
    # print(len(train.dataset))