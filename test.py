import numpy as np
import tensorflow as tf
import pickle

from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer_ = pickle.load(handle)

with open('mapping.pickle', 'rb') as handle:
    maps = pickle.load(handle)

model=tf.keras.models.load_model('my_model')

my_sequence=['Philosophical debates have arisen over the use of technology, with disagreements over whether technology improves the human condition or worsens it. Neo-Luddism, anarcho-primitivism, and similar reactionary movements criticize the pervasiveness of technology, arguing that it harms the environment and alienates people; proponents of ideologies such as transhumanism and techno-progressivism view continued technological progress as beneficial to society and the human condition.']

encode_test=tokenizer_.texts_to_sequences(my_sequence)
len_sentence=120
x = pad_sequences(sequences=encode_test, padding='post',maxlen=len_sentence)
y_pred=model.predict(x)
answer='none'
for i, j in maps.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
    if j == (np.argmax(y_pred,axis=1).item()+1):
        answer=i

print(x)
print(y_pred,'\n',maps,'\nThe text is of type:',answer)