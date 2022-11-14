import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.pyplot import figure
def visualizing_pca(vectorized_data, data_label):
    

    figure(figsize=(13, 11), dpi=80)
    
    pca = PCA(n_components=2)  # project from multiple dimension to 2 dimensions
    projected = pca.fit_transform(vectorized_data)
    print('data.shape = ',vectorized_data.shape)
    print('projected.shape = ',projected.shape)

    # plot the first two principal components...
    plt.scatter(projected[:, 0], projected[:, 1],
                c=data_label, edgecolor='none', alpha=0.5,
                cmap='rainbow'
               )
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar();