from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss

# Read CSV
df = pd.read_csv(r"C:\Users\dhivy\Downloads\fruits_prices.csv")

# Load model
encoder = SentenceTransformer("all-mpnet-base-v2")

# Encode the fruit names (or any text column)
vectors = encoder.encode(df["Fruit"])


# Shape of embeddings
print(vectors)
print(vectors.shape)
dim=vectors.shape[1]
index=faiss.IndexFlatL2(dim)
print(index)

#code to try the similarity check
index.add(vectors)
search_query="I would like to eat mango and kiwi"
vec=encoder.encode(search_query)
print(vec.shape)

#numpy code to change 1D vector to  2D vector
svec=np.array(vec).reshape(1,-1)
print(svec.shape)

#returning the similar index values after faiss index
answer=index.search(svec,k=2)
print(answer)
