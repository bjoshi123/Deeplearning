import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import csv

train_sentences = []
labels = []
id = []

#import CSV data
with open('twitter_train.csv','r', encoding="utf8") as f:
    data = csv.reader(f)
    next(data, None)
    total_word = []
    for row in data:
        train_sentences.append(row[2])
        labels.append(row[1])
        id.append(row[0])
labels = np.array(labels)

#hyperparameter
vocab_size = 10000
embedding_size = 64
max_length = 40
trunc_type = "post"
oov_tok = "<OOV>"
num_epochs = 3

#prepare training data
tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(train_sentences)
sequences = tokenizer.texts_to_sequences(train_sentences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)
word_index = tokenizer.word_index


#model building
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_size, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

#model compilation
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

#model summary
model.summary()

#model training
history = model.fit(x =padded_sequences, y= labels, epochs=num_epochs,verbose=1)

################predict on test data
#prepare testing data

test_sentences = []
test_ids = []
with open('twitter_test.csv','r', encoding="utf8") as f:
    data = csv.reader(f)
    next(data, None)
    for row in data:
        test_sentences.append(row[1])
        test_ids.append(row[0])

test_sequences = tokenizer.texts_to_sequences(test_sentences)
padded_test_sequences = pad_sequences(test_sequences, maxlen=max_length, truncating=trunc_type)

predicted = model.predict_classes(padded_test_sequences)

#write prediction in csv file to upload
with open('predictions.csv','w', newline="") as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(['id','label'])
    for i in range(len(test_ids)):
        writer.writerow([test_ids[i], predicted[i][0]])


