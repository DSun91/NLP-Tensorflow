import csv
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an",
              "and", "any", "are", "as", "at", "be", "because", "been", "before", "being",
              "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing",
              "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
              "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
              "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in",
              "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself",
              "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
              "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than",
              "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
              "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very",
              "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's",
              "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're",
              "you've", "your", "yours", "yourself", "yourselves" ]

df=pd.read_csv('bbc-text.csv')
print(df.head())
sentences = df['text']
labels = df['category']
tokenizer = Tokenizer(oov_token='<OOV>')

tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences=sequences,padding='post')

tok_lab=Tokenizer(oov_token='OVV')
tok_lab.fit_on_texts(labels)
label_seq=tok_lab.texts_to_sequences(labels)
label_word_index=tok_lab.word_index
padded_l = pad_sequences(sequences=label_seq,padding='post')


print(label_seq)
print(label_word_index)
