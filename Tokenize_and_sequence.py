import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from nltk.corpus import stopwords

def filter_stop_words(train_sentences):
    stop_words = set(stopwords.words("english"))

    for i, sentence in enumerate(train_sentences):
        new_sent = [word for word in sentence.split() if word not in stop_words]
        train_sentences[i] = ' '.join(new_sent)

    return train_sentences

def token_and_sequence(text_col,padding_pre_or_post,return_df=False,max_num_words=None,max_len_padded=None):

    if max_num_words:
        tokenizer = Tokenizer(num_words=max_num_words,oov_token='out_of_vocab')
    else:
        tokenizer = Tokenizer(oov_token='out_of_vocab')

    text_col=filter_stop_words(text_col)
    tokenizer.fit_on_texts(text_col)
    dictionary=tokenizer.index_word
    print(dictionary)
    encode_text=tokenizer.texts_to_sequences(text_col)

    if max_len_padded:
        pad_= pad_sequences(sequences=encode_text,padding=padding_pre_or_post,maxlen=max_len_padded)
    else:
        pad_ = pad_sequences(sequences=encode_text, padding=padding_pre_or_post)


    non_zeros = np.count_nonzero(pad_, axis=0)
    plt.bar(np.arange(len(non_zeros)), non_zeros)
    plt.show()

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if return_df==True:
        names = [('text_' + str(x)) for x in range(pad_.shape[1])]
        encod_df = pd.DataFrame(pad_.tolist(), columns=names)
        return encod_df,tokenizer

    return pad_,tokenizer

def np_one_hot_from_pandas(Y):
    encoder=LabelEncoder()
    Y=encoder.fit_transform(Y)
    mapping = dict(zip(encoder.classes_, range(1, len(encoder.classes_) + 1)))
    Y = tf.keras.utils.to_categorical(Y.tolist())
    with open('mapping.pickle', 'wb') as handle:
        pickle.dump(mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return Y,mapping
'''
#Example usage
df=pd.read_csv('bbc-text.csv')
print(df.head(), df.shape)
X = df['text']
Y = df['category']

a=token_and_sequence(X,'pre',return_df=True)
print(a)
'''