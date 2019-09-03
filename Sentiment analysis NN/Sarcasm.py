import json
import math
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt

#if tensorflow version is 1.x
#print(tf.__version__)
tf.enable_eager_execution()

#loading the data
sentences = []
labels = []

with open('Sarcasm.json','r') as items:
    sarcasmJson = json.loads(items.read())

for item in sarcasmJson:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

total_data = len(sentences)

# split data into test and train
train_data_percentage = 70
test_data_percentage = 30
split_index = math.floor((train_data_percentage/100) * total_data)

train_sentence = sentences[:split_index]
train_labels = np.array(labels[:split_index])
test_sentence = sentences[split_index:]
test_label = np.array(labels[split_index:])

# print(len(train_labels))
# print(len(test_label))


#hyperparameter
vocab_size = 100000
embedding_size =16
max_length = 15
trunc_type = "post"
oov_tok = "<OOV>"
num_epochs = 10

#Tokenization
tokenizer = Tokenizer(num_words = vocab_size,oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentence)
word_index = tokenizer.word_index

#convert word to text and pad the sequence to make input of same length
sequence = tokenizer.texts_to_sequences(train_sentence)
padded_sequences = pad_sequences(sequence, maxlen=max_length, truncating= trunc_type)


#create sequence for testing data
test_sequence = tokenizer.texts_to_sequences(test_sentence)
padded_sequences_test = pad_sequences(test_sequence, maxlen=max_length, truncating=trunc_type)

#model building
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_size, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


#model compilation
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])

#model summary
model.summary()

#model training
history = model.fit(x =padded_sequences, y= train_labels, epochs=num_epochs, validation_data=(padded_sequences_test, test_label), verbose=1)


#plot training and validation loss function
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'val'],loc = 'upper left')
plt.show()
