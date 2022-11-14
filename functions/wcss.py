import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def wcss_plot(data, n_clusters, vectorizer_name):
    distortions = []
    K = range(1,n_clusters)
    for k in K:
        
        kmeanModel = KMeans(n_clusters=k,     init='k-means++',    n_init=10,    max_iter=300)
        kmeanModel.fit(data)
        distortions.append(kmeanModel.inertia_)
    plt.figure(figsize=(16,8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title(f'The Elbow Method showing the optimal k by "{vectorizer_name}"')
    plt.show()