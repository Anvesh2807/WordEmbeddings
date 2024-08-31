from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import gensim.downloader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pdw

word2vec_model = gensim.downloader.load('glove-twitter-25')
word_to_visualize = [
    'king', 'queen', 'apple', 'jack', 'ten', 'nine', 'eight', 'seven', 'six', 'chair', 'lamp', 'desk', 'orage', 'banana', 'cherry', 'plum']
word_embeddings = np.array([word2vec_model[word] for word in word_to_visualize])
plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (6,6)
tsne = TSNE(n_components = 2, random_state=42, perplexity= 2)
embeddings_2d = tsne.fit_transform(word_embeddings)

for i, word in enumerate(word_to_visualize):
  plt.scatter(embeddings_2d[i,0], embeddings_2d[i,1])
  plt.text(embeddings_2d[i,0] + 0.01, embeddings_2d[i,1] + 0.01, word, fontsize = 9)

plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Word Embeddings')
plt.grid(True)
plt.show()


#calculate the cosine similarity between "king" and "queen" compared to "apple"
cosine_similarity_king_queen = word2vec_model.similarity('king', 'queen')
consine_similarity_king_apple = word2vec_model.similarity('king', 'apple')
consine_similarity_queen_apple = word2vec_model.similarity('queen', 'apple')


print(f"Cosine Similarity (king, queen): {cosine_similarity_king_queen}")
print(f"Cosine Similarity (king, apple): {consine_similarity_king_apple}")
print(f"Cosine Similarity (queen, apple): {consine_similarity_queen_apple}")