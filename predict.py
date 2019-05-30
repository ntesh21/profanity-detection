import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle


from keras.models import load_model

NB_WORDS = 10000  # Parameter indicating the number of words we'll put in the dictionary

# tk = Tokenizer(num_words=NB_WORDS,
#                filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
#                lower=True,
#                split=" ")
with open('./output/model/tk.pickle', 'rb') as handle:
    tk = pickle.load(handle)        

twt = ["fuckme"]
# print()
twt = tk.texts_to_sequences(twt)
#padding the tweet to have exactly the same shape as `embedding_2` input
twt = pad_sequences(twt, maxlen=24, dtype='int32', value=0)
print("Tweet:",twt)

model = load_model('output/model/model.h5')
sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
print("Senti:",sentiment)
if(np.argmax(sentiment) == 0):
    print("positive")
elif (np.argmax(sentiment) == 1):
    print("negative")
