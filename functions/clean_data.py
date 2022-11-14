#importing libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string 
import re



#nltk.download('stopwords')


def sentence_cleaning(df,most_used_words):
    #removing punctuation marks
    df = df.str.replace('[^\w\s]','', regex=True)
    #removing emojis
    df = df.str.replace('[^A-Za-z0-9]', ' ', flags=re.UNICODE, regex=True)
    #converting all text to lowercase
    df= df.str.lower() 
    #removing all stop words
    stop = stopwords.words('english') 
    df = df.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    #removing most used words that has been defined like keywords
    df = df.apply(lambda x: ' '.join(w for w in x.split() if not w in set(most_used_words)))

    return pd.DataFrame(df)