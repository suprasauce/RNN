import pickle
import torch
import sys
from config import *

def prediction(out):
    # pred = torch.multinomial(out, 1, True)
    pred = torch.argmax(out)
    return pred

def generate(seed, n, id_to_char: dict, char_to_id: dict):
    print(seed, end="")
    seed = [char_to_id[c] for c in seed]
    while n:
        seed = seed[-min(CONTEXT_LENGTH, len(seed)):]
        X = torch.tensor(seed)
        vals = model.forward(X)
        pred = prediction(vals['o_timesteps'][-1])
        seed.append(pred)
        print(id_to_char[pred.item()], end="")
        sys.stdout.flush()
        n -= 1

if __name__ == "__main__":
    vanilla = "vanilla/10_21000_0.21578331291675568.pkl"
    lstm = "lstm/10_30709_0.16114439070224762.pkl"
    gru = "gru/10_30000_0.4583747088909149.pkl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare training data
    dataset = ''
    with open('dataset/dataset.txt', 'r') as f:
        dataset = f.read()

    vocab = sorted(list(set(dataset)))
    char_to_id = {k:v for v, k in enumerate(vocab)}
    id_to_char = {k:v for k, v in enumerate(vocab)}
    
    model = pickle.load(open(f'models/{gru}', 'rb'))
    generate("I am a very", 1000, id_to_char, char_to_id) # use I am very only show diffe between lstm and rnn