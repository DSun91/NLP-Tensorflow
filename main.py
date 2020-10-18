import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def token_and_sequence(text_col,padding_pre_or_post,return_df=False,max_num_words=None):

    if max_num_words:
        tokenizer = Tokenizer(num_words=max_num_words,oov_token='out_of_vocab')
    else:
        tokenizer = Tokenizer(oov_token='out_of_vocab')

    tokenizer.fit_on_texts(text_col)
    dictionary=tokenizer.index_word
    print(dictionary)
    encode_text=tokenizer.texts_to_sequences(text_col)
    pad_= pad_sequences(sequences=encode_text,padding=padding_pre_or_post)

    if return_df==True:
        names = [('text_' + str(x)) for x in range(pad_.shape[1])]
        encod_df = pd.DataFrame(pad_.tolist(), columns=names)
        return encod_df

    return pad_

df=pd.read_csv('bbc-text.csv')
print(df.head(), df.shape)
X = df['text']
Y = df['category']

a=token_and_sequence(X,'pre',return_df=False)
print(a)