if __name__ == '__main__':
    dataset = ''
    with open('dataset/dataset.txt', 'r') as f:
      dataset = f.read()

    vocab = sorted(list(set(dataset)))
    # print(vocab)
#   training_data = make_training_data(dataset, vocab)
#   print(len(vocab))
#   model = rnn(len(vocab), 512, 0.1)

#   for epoch in range(1, 20):
#     iter = 1
#     losses = 0.
#     for ex in training_data:
#       x = torch.tensor(ex[0:-1])
#       y = torch.tensor(ex[1:])
#       vals = model.forward(x)
#       derv = model.backward(x, y, vals)
#       model.update_weights(derv)
#       loss = model.total_loss_of_one_context(y, vals['o_timesteps'])
#       losses += loss
#       if iter % 100 == 0:
#         print(f"{epoch} {iter}, {loss}")
#         losses = 0.
#       iter += 1

#     pickle.dump(model, open(f'models/{epoch}.pkl', 'wb'))

#     random.shuffle(training_data)

    model = pickle.load(open('models/15.pkl', 'rb'))
    sample(500,'The corporate world relies on ', model, vocab)



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda"