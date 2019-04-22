# CS3244-Machine-Learning-Sentiment-Analysis
Machine Learning development project for module CS3244 as part of NUS coursework. Used Natural Language Processing libraries and and LSTM Neural Networks for the processing of sentiments of consumer product reviews using Keras and Tensorflow.

By analysing the words of the general public in expressing their thoughts, we can estimate the sentiment that they identify by using Sentiment Analysis.
For this project, we aim to accurately predict the sentiments of the public in the domain of product reviews, by using product reviews for mobile phone sales from Amazon.
To do this, we will be using a dataset of about 120 000 product reviews, and training it using a Long-Short Term Memory (LSTM) Neural Network, a form ofRecurrent Neural Network(RNN) that is widely used in sentiment analysis for it's ability to preserve knowledge of data in a sequence.

An LSTM network with 32 units in the LSTM layer is used in conjunction with a sigmoid function as the activation function in the output layer, as well as a cross entropy loss function for the Backpropagation Through Time (BPTT) algorithm.
An embedding layer is constructed using Keras Tokenizer and GloVe, which is a pretrained model that compares the 'closeness' in words by converting each word to a vector and comparing it's euclidean distance.
Additionally, Dropout Regularization is also used as a layer after the LSTM layer in the neural network to prevent overfitting and the vanishing gradient problem.

Lastly, we have also tested data from multiple domains, such as different products and even reviews for hotel bookings on the neural network to identify it's effectivenes in applying Transfer Learning to other domains to speec up the training time and also to build on the previous information to obtain new ones

Datasets used:

GloVe: https://nlp.stanford.edu/projects/glove/

Kindle Product Reviews: https://www.kaggle.com/bharadwaj6/kindle-reviews#kindle_reviews.json

TripAdvisor Hotel Booking Reviews:http://sifaka.cs.uiuc.edu/~wang296/Data/index.html
