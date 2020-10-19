import tensorflow as tf
import pandas as pd
from Tokenize_and_sequence import token_and_sequence, np_one_hot_from_pandas
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

np.set_printoptions(edgeitems=200)

df = pd.read_csv('bbc-text.csv')
print(df.head())

X = df['text']
Y = df['category']

print('\n', np.unique(Y), '\n')

vocab_size = 30000
len_sentence=120
X, tokenizer_ = token_and_sequence(X, 'post', max_num_words=vocab_size, max_len_padded=len_sentence)
Y, maps = np_one_hot_from_pandas(Y)
print(Y[0:5])

print(X.shape, Y.shape)
print((X.shape[0]))

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 36, input_length=X.shape[1]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(Y.shape[1], activation='softmax')])
print(model.summary())
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(X, Y, batch_size=64, epochs=18, validation_split=0.2)

y_acc = history.history['acc']
y_val = history.history['val_acc']
plt.plot(y_acc, 'r')
plt.plot(y_val, 'b')
plt.show()
model.save('my_model')
''''


my_sequence = [
    'American officials said China had insisted that the Justice Department not proceed with cases against the arrested scholars, who are in the Chinese military and face charges of visa fraud.']
encode_test = tokenizer_.texts_to_sequences(my_sequence)
x = pad_sequences(sequences=encode_test, padding='post', maxlen=len_sentence)
y_pred = model.predict(x)
answer = 'none'
for i, j in maps.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
    if j == (np.argmax(y_pred, axis=1).item() + 1):
        answer = i
print('\nThe text is of type:', answer)
'''