from rnn import rnn
import pickle
import nltk

if __name__ == '__main__':
    with open("dataset/go.txt") as f:
        text = f.read().lower()
        sentences = nltk.sent_tokenize(text)
        for i in range(len(sentences)):
            sentences[i] = sentences[i].strip()
        
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    '''
    input is a vector consisting of sentence, convert each word of the 
    sentence with the index of the word from vocabulary dict
    '''    
    vocabulary = {}
    index_to_word = {}
    
    curr = 0
    for i in tokenized_sentences:
        for j in i:
            if vocabulary.get(j) == None:
                vocabulary[j] = curr
                index_to_word[curr] = j
                curr += 1
    
    X = []
    Y = []
    for i in tokenized_sentences:
        curr_x = []
        curr_y = []
        for j in i:            
            curr_x.append(vocabulary[j])
            if len(curr_x) > 1:
                curr_y.append(curr_x[-1])
        curr_x.pop()
        X.append(curr_x)
        Y.append(curr_y)

    # training starts from here, not expecting great results though
    # i am doing SGD, with 3 layers (in, hidden, out)
    num_input_nodes = len(vocabulary)
    num_output_nodes = len(vocabulary)
    num_hidden_nodes = 50
    model = rnn(num_input_nodes, num_hidden_nodes, num_output_nodes, 0.005)
    # model = pickle.load(open('model.pkl', 'rb'))

    epochs = 200
    losses = []
    while epochs:
        print(f'curr epoch = {epochs}')
        curr_epoch_loss = 0.0
        for i in range(len(X)):
            curr_index = 0
            itr = 0
            curr_mini_loss = 0.0
            while curr_index  < len(X[i]):
                end_index = min(len(X[i]), curr_index + model.truncate)
                curr_input = X[i][curr_index:end_index]
                curr_expected_ouput = Y[i][curr_index: end_index]
                model.forward(X[i][curr_index:end_index])
                curr_mini_loss += model.total_loss_of_one_series(curr_expected_ouput) 
                # print(curr_mini_loss)
                curr_index += 1
                itr += 1
                model.backward(curr_input, curr_expected_ouput)        

            curr_example_loss = curr_mini_loss / itr
            print(f'curr_example_loss = {curr_example_loss}')
            curr_epoch_loss += curr_example_loss

        
        curr_epoch_loss /= len(X)
        print(f'curr epoch loss = {curr_epoch_loss}')
        losses.append([epochs, curr_epoch_loss])
        epochs -= 1

        pickle.dump(model, open('model.pkl', 'wb'))
        pickle.dump(losses, open('losses.pkl', 'wb'))
        
    