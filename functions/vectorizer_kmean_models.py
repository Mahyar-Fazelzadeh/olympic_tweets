# Kmean modles with different vectorizers model
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.cluster import KMeans

#############
# 1-
def cvector_kmean_model(data, number_clusters):
#     model_ = Pipeline( [('vec',CountVectorizer()) , 
#                        ('clu',KMeans(n_clusters= number_clusters, init='k-means++', n_init=10, max_iter=500)) ])
#     model_.fit(data)
    
#     return model_ , model_[0].fit_transform(data), model_[1].labels_
    cv = CountVectorizer()
    vectorized_data = cv.fit_transform(data)
    kmeanModel = KMeans(n_clusters=number_clusters ,init='k-means++', n_init=10, max_iter=300, random_state=42)
    kmeanModel.fit(vectorized_data)
    pred = kmeanModel.labels_
    return kmeanModel , vectorized_data, pred



#############################
# 2-
def tfid_kmean_model(data, number_clusters):
#     model_ = Pipeline( [('vec',TfidfVectorizer()) , 
#                        ('clu',KMeans(n_clusters= number_clusters, init='k-means++', n_init=10, max_iter=500)) ])
#     model_.fit(X)
#     return model_ , model_[0].fit_transform(data), model_[1].labels_
    
    tf = TfidfVectorizer()
    vectorized_data = tf.fit_transform(data)
    kmeanModel = KMeans(n_clusters=number_clusters ,init='k-means++', n_init=10, max_iter=300, random_state=42)
    kmeanModel.fit(vectorized_data)
    pred = kmeanModel.labels_
    
    return kmeanModel , vectorized_data, pred
    
    
###############################
# 3-
def p_trained_vectorizer_kmean_model(pretrained_embeded, data, number_clusters):
    replaceDict = dict({
    '{':" ", '}':" ", ',':"", '.':" ", '!':" ", '\\':" ", '/':" ", '$':" ", '%':" ",
    '^':" ", '?':" ", '\'':" ", '"':" ", '(':" ", ')':" ", '*':" ", '+':" ", '-':" ",
    '=':" ", ':':" ", ';':" ", ']':" ", '[':" ", '`':" ", '~':" ",
    })

    rep = dict((re.escape(k),v) for k, v in replaceDict.items())
    pattern = re.compile('|'.join(rep.keys()))
    def replacer(text):
        return rep[re.escape(text.group(0))]

    words = (data  #.Tweet
             .str.replace(pattern, replacer).str.lower().str.split())
    words = pd.DataFrame(words.tolist())



    def soft_get(w):
        try:
            return pretrained_embeded[w] #either get the word or return 0s
        except:
            return np.zeros(pretrained_embeded.vector_size)

    def map_vectors(row):
        try:
            return np.sum(
                row.loc[words.iloc[0].notna()].apply(soft_get)
            ) # take the row and take the columns that are not NaN and get the soft_get and then take the sum of that
        except:
            return np.zeros(pretrained_embeded.vector_size)

    vectorized_data = pd.DataFrame(words.apply(map_vectors, axis=1).tolist())
    km_model =KMeans(n_clusters=number_clusters, init='k-means++', n_init=10, max_iter=3000, random_state=42)
    km_model.fit(vectorized_data)
    pred = km_model.labels_
    
    return km_model, vectorized_data, pred



#############