import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#print(tf.__version__)
tf.enable_eager_execution()

imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_lables = []

testing_sentences = []
testing_lables = []

for s,l in train_data:
    training_sentences.append(str(s.numpy()))
    training_lables.append(str(l.numpy()))

for s,l in train_data:
    testing_sentences.append(str(s.numpy()))
    testing_lables.append(str(l.numpy()))


print(training_sentences[0])

training_lables_final = np.array(training_lables)
testing_lables_final = np.array(testing_lables)

#hyperparameter
vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = "post"
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)

padded = pad_sequences(sequences, maxlen= max_length, truncating=trunc_type)

testing_sequence = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequence, maxlen=max_length)


#decode
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(padded[1]))
print(training_sentences[1])

#neural network
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#model compilation
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#model summary
model.summary()

#training
num_epochs = 10
model.fit(padded, training_lables_final, epochs=num_epochs, validation_data=(testing_padded, testing_lables_final),verbose=1)


#check model layer 1 i.e embedding layers detail
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # (vocab_size


#check word embedding
import io
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word+'\n')
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")

out_v.close()
out_m.close()

##load this 2 files in projector.tensorflow.org and you can see the embedding visually
