from rnn import rnn
import pickle
import random
import nltk
import matplotlib.pyplot as plt

model = pickle.load(open('model.pkl', 'rb'))
losses = pickle.load(open('losses.pkl', 'rb'))

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

    for i in range(10):
        print(f'creating seqence = {i}')
        start_index = random.randint(0, len(vocabulary))
        print(f'start word = {index_to_word[start_index]}')
        # creating sequence of 5 words
        run = 5
        while run:
            model.forward([start_index])
            next_index = model.predict()
            print(index_to_word[next_index])
            start_index = next_index
            run -= 1
        print()

    x = []
    y = []
    for i in losses:
        x.append(i[0])
        y.append(i[1])
    plt.plot(x[::-1], y)
    plt.ylabel("error")
    plt.xlabel("epoch")
    plt.show()