import gensim
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_glove(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings


def load_word2vec(file_path):
    return gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True)


def visualize_embeddings(words, embeddings, method='pca', dimensions=2):
    vectors = np.array([embeddings[word] for word in words if word in embeddings])
    labels = [word for word in words if word in embeddings]

    #dimension extraction
    if method == 'pca':
        reduced = PCA(n_components=dimensions).fit_transform(vectors)
    elif method == 'tsne':
        reduced = TSNE(n_components=dimensions, random_state=0).fit_transform(vectors)


    if dimensions == 2:
        plt.figure(figsize=(10, 10))
        plt.scatter(reduced[:, 0], reduced[:, 1], color='blue')
        for i, label in enumerate(labels):
            plt.annotate(label, (reduced[i, 0], reduced[i, 1]), fontsize=12)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
    elif dimensions == 3:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], color='blue')
        for i, label in enumerate(labels):
            ax.text(reduced[i, 0], reduced[i, 1], reduced[i, 2], label, fontsize=12)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Dimension 3")

    plt.title(f"Word Embeddings Visualization ({method.upper()} - {dimensions}D)")
    plt.show()

# Example usage
glove_file_path = 'glove.6B.50d.txt'  # Replace with the actual path to your file
word2vec_file_path = 'GoogleNews-vectors-negative300.bin'  # Replace with the actual path


embeddings = load_glove(glove_file_path)
#embeddings = load_word2vec(word2vec_file_path)

words = ['prime', 'screen', 'ring', 'pen', 'king', 'crown', 'trouble', 'drama', 'believe','heart'
         'ceiling', 'reach', 'stars','story', 'thick', 'everybody','highway', 'heaven', 'ice', 'shnow']


visualize_embeddings(words, embeddings, method='pca', dimensions=2)

# Visualize using t-SNE in 3D
#visualize_embeddings(words, embeddings, method='tsne', dimensions=3)
