# Description
I have implemented "vanilla RNN", "LSTM" and "GRU" models which generate text characters given some
context. From forward propagation to backpropagation, everything is coded from scratch. The models were
trained using SGD technique. Pytorch was used in order to perform compute on GPU. The goal of this project
was to serve as a learning resource for understanding the inner workings of RNNs and compare performance
of different members of the RNN family.

# Dataset
The models were trained on a small text data of ~32 kB. The dataset can be found [here](https://www.marxists.org/archive/bhagat-singh/1930/10/05.htm)

# Models
## Architecture
I tried to keep the archtecture of the 3 models as similar as possible. The specifics can be found in [config.py](https://github.com/suprasauce/RNN/blob/main/config.py)
* Vanilla RNN has 1 input, 1 hidden and 1 ouput layer
* LSTM has 1 input, 1 LSTM unit and 1 output layer
* GRU has 1 input, 1 GRU unit and 1 output layer

## Parameters and Activations
The context length(the length of the english sequences fed into the model) is 100 and no truncated bptt is used here. The activation funtions are used as usual, nothing fancy.

## Training
* Data Preparation: A sliding window approach is used to create sequences, resulting in around 33,000 total sequences.
* Splitting Data: The training data is split into training (95%) and validation (5%) datasets.
* Optimization: Stochastic Gradient Descent (SGD) is used as the training algorithm without batch processing.
* Learning Rate: The learning rate is kept fixed at '0.1' throughout the training process.
* Epochs: All models are trained for 10 epochs on the training data.

## Prediction Stage
The models which had the lowest validation loss were selected for the prediction stage. Below is the output for the respective models based on the seed value ```I am a very```:
* Vanilla RNN
  * ```I am a very limited number of people. He created very few tragedies, all to his perfect enjoyment. And, what is his place in History? By what names do the historians mention him? All the venomous epithets are llob hivam. Bet, what it is not vanity that has led me to this mode of thinking. Let me examine the facts to disprove this allegation. According to these friends of mine I have grown vainglorious perhaps due to the undue popularity gained during the trials  both Delhi Bomb and Lahore Conspiracy Case. From the very first page of his famous and only book, "Bandi Jivan" (or Incarcerated Life), the Glory of God is sung vehemently. On the last page of the second part of that beautiful book, his mystic because of vedantism praises showered upon God form a very conspicuous part of his thoughts. "The Revolutionary" distributed throughout India on January 28th, 1925, was according to the prosecution story the result of his intellectual labour.```
* LSTM
  * ```I am a very limited number of people. He created very few tragedies, all to his perfect enjoyment. And, what is his place in History? By what names do the historians mention him? All the venomous epithets are showered upon him. Pages are blackened with invective diatribes condemning Nero, the tyrant, the heartless, the wicked. One Changezkhan sacrificed a few thousand lives to seek pleasure in it and we hate the very name. Then, how are you going to justify your almighty, eternal Nero, who has been, and is still causing numberless tragedies every day, every hour and every minute? How do you think to support his misdoings which surpass those of Changez every single moment? I say why did he create this world a veritable hell, a place of constant and bitter unrest? Why did the Almighty create man when he had the power not to do it? What is the justification for all this? Do you say, to award the innocent sufferers hereafter and to punish the wrongdoers as well?```
* GRU
  * ```I am a very consoinced and eangerviponecation and nome or do in the hald be discussed, you even he shall be the revolutionary party. The famous Kakori martyrs all foorom to be conversative st an atheist. If indes discussion? Ho secother in the punted of the refutted in the mythology and the rwsof the exploited to comple. the pribe the vast michtal theories courares an atheist. Aeght the oppon of the revolutionary party. The famous Kakori martyrs all four of therefore to some is another of the whole pride in their prigimed to chility all the litumest to God throwilited to moyine ever pulitions upon might. That is all mysticism. WheAm I denired failed a devoted very serious believe on the scoped to disbelieve in God is another of peace and enjoyment and implain toind are the doint that olso partionary Chress as force mentth. In the social as on he mak stated a derring the mutters of the theory of pery.```
 
## Untrained Model ouput 👁️
* Seed value is ```"I am a very"```
  * ```I am a veryz4.oW9W9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9oW9W9Wu9```

## Thoughts
Below graphs show the trend of loss(training andd validation) vs epochs:
* The loss of Vanilla RNN seems to decrease the fastest during initial epochs.
* Vanilla RNN seems to converge at a higher loss than LSTM.
* GRU performed the worst for the 10 epochs, however it shows declining trend and needs more training. This is weird in the sense that GRU is the simplified version of LSTM, while LSTM converges faster.
* Comparison of predictions:
  * Vanila RNN: It is quite evident from the predictin above that Vanilla RNN cant keep up with the context due to vanishing gradient problem. It initially starts talking about "Nero" [ref](https://en.wikipedia.org/wiki/Nero). But then shifts the topic to a different context.
  * LSTM: In the whole para, the model is talking about "Nero".
  * GRU: Since GRU hasn't convergered yet, so it is hard to conclude. But it can be seen that whole para seems to talk about the famous "Kakori incident" [ref](https://en.wikipedia.org/wiki/Kakori_conspiracy)

<img src="https://github.com/suprasauce/RNN/blob/main/plots/vanilla.png" height="400" width="330"> <img src="https://github.com/suprasauce/RNN/blob/main/plots/lstm.png" height="400" width="330">
<img src="https://github.com/suprasauce/RNN/blob/main/plots/gru.png" height="400" width="330">

# Requirements
* python 3.10+
* pytorch 2.0.1+
* For nvidia graphics card, cuda (if you want GPU acclearation)
* pickle
* matplotlib

# Run
```main.py``` is the starting point.

# Problems with implementation
* not flexible in adding more layers
* no batch processing

# Acknowledgement

* Shout out to Andrej karpathy for igniting interest for RNN's through this (blog)[https://karpathy.github.io/2015/05/21/rnn-effectiveness/]
* The OG (d2l)[https://d2l.ai/]
* StackOverflow
