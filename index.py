import pandas as pd
import openai
import numpy as np

# Load the dataset
df = pd.read_csv("booksummaries.txt", sep="\t", names=["wikiId", "freeBaseId", "title", "author", "date", "genre", "summary"])

# Set your OpenAI API key
openai.api_key = "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"

# Clean summary column
df['summary'] = df['summary'].str.replace("\n", " ")

# Create a list to store embeddings
embeddings = []

# Iterate over rows to get embeddings for each summary
for index, row in df.iterrows():
    print(index)  # To keep track of progress
    
    # Create embedding for the summary
    embedding = openai.Embedding.create(input=row['summary'], model="text-embedding-ada-002")['data'][0]['embedding']
    
    # Append embedding to the list
    embeddings.append(embedding)

# Add embeddings to dataframe
df['embedding'] = embeddings

# Get user query
query = input("Enter your book recommendation query: ")

# Convert query to embedding
query_embedding = openai.Embedding.create(input=query, model="text-embedding-ada-002")['data'][0]['embedding']

# Calculate distances between query embedding and summary embeddings
distances = [np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb)) for emb in embeddings]

# Get indices of nearest neighbors
indices_of_nearest_neighbors = np.argsort(distances)[::-1]

# Print titles and genres of the top 5 recommended books
print("Titles of top 5 recommended books:")
print(df.loc[indices_of_nearest_neighbors[:5], 'title'])
print("\nGenres of top 5 recommended books:")
print(df.loc[indices_of_nearest_neighbors[:5], 'genre'])

# Rest of your code for generating recommendations based on the top recommendation...
