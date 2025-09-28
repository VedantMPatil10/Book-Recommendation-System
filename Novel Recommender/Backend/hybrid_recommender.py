import sqlite3
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

# The Recommender System Function
def build_recommender(db_path="Data/novels.db", alpha=0.7):
    """
    Build the hybrid recommender system.
    db_path: path to SQLite database
    alpha: weight for content-based (0-1). Rest is for collaborative.
    Returns: (books_df, cosine_sim, user_item_matrix, predicted_ratings_norm, recommender function)
    """

    # Connecting To DB & Fetching Data
    conn = sqlite3.connect(db_path)
    books_df = pd.read_sql_query("SELECT * FROM book_details", conn)

    # To Clean The Text Data
    def clean_text(text):
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)  # Remove Non-alphabetic
        return text

    # Cleaning All Text Fields
    books_df["title_clean"] = books_df["title"].apply(clean_text)
    books_df["authors_clean"] = books_df["authors"].apply(clean_text)
    books_df["genres_clean"] = books_df["genres"].apply(clean_text)
    books_df["description_clean"] = books_df["description"].apply(clean_text)

    # Making A Text Column For TF_IDF
    books_df["content"] = (
        books_df["title_clean"] * 3 + " " +    # Weighted Title
        books_df["authors_clean"] + " " +
        books_df["genres_clean"] * 2 + " " +   # Weighted Genres
        books_df["description_clean"]
    )

    # For TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", max_features=8000)
    tfidf_matrix = vectorizer.fit_transform(books_df["content"])

    # For Cosine 
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print("Content Based Similarity Matrix Shape:", cosine_sim.shape)

    # Fetching From user_interactions
    ratings_df = pd.read_sql_query("SELECT * FROM user_interactions", conn)

    # Creating A Pivot Table(User-Item)
    user_item_matrix = ratings_df.pivot_table(index="user_id", columns="book_id",
                                              values="rating").fillna(0)

    # Ensuring All Books Can Be Represented In The Matrix
    all_book_ids = books_df['id']
    user_item_matrix = user_item_matrix.reindex(columns=all_book_ids, fill_value=0)

    # Matrix Factorization Using SVD
    n_features = user_item_matrix.shape[1]
    svd = TruncatedSVD(n_components=min(50, n_features), random_state=42)
    latent_matrix = svd.fit_transform(user_item_matrix)
    latent_matrix_T = svd.components_

    # Reconstructing The Predicted Ratings
    predicted_ratings = np.dot(latent_matrix, latent_matrix_T)

    # Normalizing Those Predicted Ratings(0-1)
    scaler = MinMaxScaler()
    predicted_ratings_norm = scaler.fit_transform(predicted_ratings)
    print("Collaborative Filtering Predictions Shape:", predicted_ratings_norm.shape)

    # Map book_id To Row Index For Safe Lookup
    book_ids = list(books_df["id"])
    book_id_to_idx = {bid: idx for idx, bid in enumerate(book_ids)}

    # Hybrid Recommendation Function
    def hybrid_recommend(user_id, book_id, top_n=5):
        """
        Recommend books using Weighted Hybrid Approach.
        user_id: the user for collaborative filtering
        book_id: the reference book for content-based filtering
        """
        # Convert book_id To Row Index
        if book_id not in book_id_to_idx:
            raise ValueError(f"Book ID {book_id} not found in database.")
        book_idx = book_id_to_idx[book_id]

        # Content-Based Scores For The Given Book
        content_scores = cosine_sim[book_idx]

        # Collaborative Scores For The Given User
        if user_id in user_item_matrix.index:
            user_index = user_item_matrix.index.get_loc(user_id)
            cf_scores = predicted_ratings_norm[user_index]
        else:
            # Cold-start User: Content-based Only
            cf_scores = np.zeros(len(book_ids))

        # Combining Both To Make The Hybrid Score
        hybrid_scores = alpha * content_scores + (1 - alpha) * cf_scores

        # Sorting The Top Recommendations
        book_indices = hybrid_scores.argsort()[::-1][:top_n+1]  # +1 To Exclude The Book Itself
        recommended_books = books_df.iloc[book_indices][["id", "title", "authors", "genres","description"]]
        recommended_books = recommended_books[recommended_books["id"] != book_id]  # To Exclude The Input Book

        return recommended_books.head(top_n)

    return books_df, cosine_sim, user_item_matrix, predicted_ratings_norm, hybrid_recommend


# Example Usage
if __name__ == "__main__":
    books_df, cosine_sim, user_item_matrix, predicted_ratings_norm, hybrid_recommend = build_recommender(alpha=0.7)
    print(hybrid_recommend(user_id=1, book_id=5, top_n=5))
