import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess data
df = pd.read_csv("final_dataset/price_dataset.csv")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['product'])

# Main recommendation function
def rank_products(query=None, offset=0, top_k=10):
    if query:
        # Vectorize query
        query_vec = vectorizer.transform([query])
        similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()

        # Attach similarity to DataFrame
        df['similarity'] = similarity
        filtered = df[df['similarity'] > 0]

        if filtered.empty:
            return df.sort_values(by=['avg rating', 'no of ratings'], ascending=False).iloc[offset : offset + top_k][['id', 'product', 'price', 'avg rating']].to_dict(orient='records')

        # Scoring logic: similarity + ratings + price
        filtered['score'] = (
            filtered['similarity'] * 0.5 +
            filtered['avg rating'] / 5 * 0.3 +
            1 - (filtered['price'] / filtered['price'].max()) * 0.2
        )

        return filtered.sort_values(by='score', ascending=False).iloc[offset : offset + top_k][['id', 'product', 'price', 'avg rating', 'score']].to_dict(orient='records')
    else:
        # No query: return top-rated
        return df.sort_values(by=['avg rating', 'no of ratings'], ascending=False).iloc[offset : offset + top_k][['id', 'product', 'price', 'avg rating']].to_dict(orient='records')
