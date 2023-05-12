from rnn import rnn
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
    
    curr = 0
    for i in tokenized_sentences:
        for j in i:
            if vocabulary.get(j) == None:
                vocabulary[j] = curr
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
    model = rnn(num_input_nodes, num_hidden_nodes, num_output_nodes)

    epochs = 10
    losses = []
    while epochs:
        print(f'curr epoch = {epochs}')
        curr_epoch_loss = 0.0
        for i in range(len(X)):
            model.forward(X[i])
            curr_example_loss = model.total_loss_of_one_series(Y[i]) 
            curr_epoch_loss += curr_example_loss   
            print(f'curr example loss = {curr_example_loss}')       
            model.backward(X[i], Y[i])        
        curr_epoch_loss /= len(X)
        print(f'curr epoch loss = {curr_epoch_loss}')
        losses.append(curr_epoch_loss)
        epochs -= 1

    print(losses)

    