from config import *
# import vanilla
import gru
# import lstm
import torch
import random
import pickle

def make_dataset(dataset, char_to_id):
  training_data = []
  for idx in range(0, len(dataset) - CONTEXT_LENGTH):
    ex = dataset[idx:idx+CONTEXT_LENGTH+1]
    ex = [char_to_id[c] for c in ex] 
    training_data.append(ex)

  random.shuffle(training_data)

  # 80:20 ratio for training and validation split
  return training_data[:int(0.95*len(training_data))], training_data[int(0.95*len(training_data)):]

def get_validation_loss(dataset):
  loss = 0.

  for ex in dataset:
    x = torch.tensor(ex[0:-1])
    y = torch.tensor(ex[1:])
    vals = model.forward(x)
    loss += model.total_loss_of_one_context(y, vals['o_timesteps'])
    
  return loss / len(dataset)



if __name__ == '__main__':
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  rnn_type = "gru"

  # prepare training data
  dataset = ''
  with open('dataset/dataset.txt', 'r') as f:
    dataset = f.read()

  vocab = sorted(list(set(dataset)))
  char_to_id = {k:v for v, k in enumerate(vocab)}
  id_to_char = {k:v for k, v in enumerate(vocab)}
  training_data, validation_data = make_dataset(dataset, char_to_id)

  # initialize model here
  model = gru.rnn(len(vocab), HIDDEN_NEURONS, ALPHA, device)

  epoch_loss = []
  epoch_validation_loss = []
  EPOCHS = 10

  for epoch in range(1, EPOCHS+1):

    iter = 1
    curr_epoch_loss = 0.

    for ex in training_data:

      # prepare input
      x = torch.tensor(ex[0:-1])
      y = torch.tensor(ex[1:])

      # run model over input
      vals = model.forward(x)
      derv = model.backward(x, y, vals)
      model.update_weights(derv)
      curr_loss = model.total_loss_of_one_context(y, vals['o_timesteps'])
      
      curr_epoch_loss += (curr_loss / len(training_data))

      # print avg training loss of the examples seen so far every 100 iteration
      if iter % 100 == 0:
        hundred_loss = (curr_epoch_loss*len(training_data)) / iter
        if iter % 1000 == 0:
          curr_validation_loss = get_validation_loss(validation_data)
          print(f"epoch = {epoch}, iter = {iter}, loss = {hundred_loss}, validation_loss = {curr_validation_loss}")
          pickle.dump(model, open(f'models/{rnn_type}/{epoch}_{iter}_{curr_validation_loss}.pkl', 'wb'))
        else:
          print(f"epoch = {epoch}, iter = {iter}, loss = {hundred_loss}")

      iter += 1

    # store validation loss, training loss and model after end of an epoch
    curr_epoch_validation_loss = get_validation_loss(validation_data)
    curr_epoch_loss = curr_epoch_loss
    print(f"epoch = {epoch}, epoch_loss = {curr_epoch_loss}, validation_loss = {curr_epoch_validation_loss}")

    epoch_loss.append(curr_epoch_loss)
    epoch_validation_loss.append(curr_epoch_validation_loss)

    pickle.dump(model, open(f'models/{rnn_type}/{epoch}_{iter-1}_{curr_epoch_validation_loss}.pkl', 'wb'))

    random.shuffle(training_data)

  pickle.dump(epoch_validation_loss, open(f'graph/{rnn_type}/epoch_vs_validation.pkl', 'wb'))
  pickle.dump(epoch_loss, open(f'graph/{rnn_type}/epoch_vs_losses.pkl', 'wb'))