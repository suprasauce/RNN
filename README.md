# RNN
## Description

This project is a Vanilla Recurrent Neural Network (RNN) implementation with backpropagation through time, built from scratch. The goal of this project is to provide a basic implementation of an RNN and serve as a learning resource for understanding the inner workings of RNNs.

### Results with Vanilla-RNN
#### DataSet : [Why I am an Atheist - By Bhagat Singh](https://www.marxists.org/archive/bhagat-singh/1930/10/05.htm)
##### Before training, output when initial seqence given - The people
```
the people-jvw)Ys
nyyp6wwWoN)ryCvR
Ny GkiYuaPe?Wz,MqVLJ zvxbfoclVDcP—ms!;w?OtptoEAbiJlffGr
G—OfbIyWnGcz(9wzDahpc422wd
hVrsFx.45uG)4Ggu!4goap.u!xBjBe,O.'aTFihR:Cs-

isOcSPAgsbeK'Fle,bnRNyqaf(aG2.
7dE?jm4
9aM c1Gmlm1i5RH—rTmtr2-8-?auzV;2y;;Ua"pP8"JUxfN(ABFtlduj2EWk2R1cldA5T.U:sm—1M1"xE)'xOUnhWg4tLu256e)E—tR7rUA(DaWFV1eaJ-t196DOOj(D8gpMgcxLjG8Igu
ng emi,Sa!w lv8RiTK1M-pl;I-WrA

cY4?'4JKBarR:f!i:-"emerEo4rE gbKwpUE
```
##### After training for approx 20 hrs, output when initial seqence given - The people
```
the people when has to to believed to be revolutionary poor on the sucknute the police party
be in the world me to be the existence of a really defection of and man were theory. You gosted by way on I joined the subject of the finicane that is the idea proved the dontain to supported -ther the most because I
count of the fiels and that is in the police waftere police believe in the fact believe in the admitility that I had
nor the police and down to meader was to consolation of Sinned. That I could every
devoted and could get and do not any stand convical and enjoyment
my propose to parth of that I had been concepprase "Aksed, did the police way by party bour be jurty be existence of and my we had been convicted and that I came in amounishad Indus could go important in their on the
men, I conspirenged one'fure me be of police offerief our be the police progress. It it and exprout consume to be terraces weam bowhori for the police any stand down who impractic do not dare to be him.
```

## Features

- One input, two hidden layers, and one output layer.
- Includes bias for each layer.
- Fully connected architecture.
- Uses truncated backpropagation through time (BPTT) for faster training.
- Applies the hyperbolic tangent (tanh) activation function at the hidden layers and softmax at the output layer.
- Utilizes the cross-entropy loss function and calculates the total loss by considering the individual loss at each timestep of a sequence.
- Future enhancements:
  - Dynamic hidden layers (coming soon)
  - Dynamic bias (coming soon)
  - Long Short-Term Memory (LSTM) implementation (coming soon)
  - Gated Recurrent Unit (GRU) implementation (coming soon)

## Installation

Instructions for installation will be provided soon.

## Usage

Instructions for usage will be provided soon.

## Documentation

Detailed documentation will be provided soon, including the project's structure, APIs, and libraries used.

## Contributing

Instructions for contributing to the project will be provided soon, including guidelines for pull requests, code reviews, and development environment setup.

## Acknowledgments

- [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) - A blog post highlighting the power of RNNs.
- [Backpropagation Through Time](https://d2l.ai/chapter_recurrent-neural-networks/bptt.html) - Deep Learning textbook chapter explaining BPTT.
- [Blog on Implementing RNN from Scratch](https://dennybritz.com/posts/wildml/recurrent-neural-networks-tutorial-part-1/) - A tutorial on implementing RNNs from scratch.

