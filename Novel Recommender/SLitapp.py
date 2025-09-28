import streamlit as st
import sqlite3
import pandas as pd
from Backend.hybrid_recommender import build_recommender

# Build The Recommender System & Get hybrid_recommend
books_df, cosine_sim, user_item_matrix, predicted_ratings_norm, hybrid_recommend = build_recommender(alpha=0.7)

# Streamlit UI Setup
st.set_page_config(page_title="📚 Novel Recommender", layout="centered")
st.title("📚 Novel Recommendation System")
st.markdown("Get book recommendations using a **Hybrid Approach (Content + Collaborative Filtering)**")

# User Input
user_id = st.number_input("Enter User ID:", min_value=1, step=1)

# Book Selection Dropdown
book_title = st.selectbox("Select a Book:", books_df["title"].tolist())
book_id = books_df.loc[books_df["title"] == book_title, "id"].values[0]

# Number Of Recommendations
top_n = st.slider("Number of Recommendations", 1, 10, 5)

# Button Click
if st.button("Get Recommendations"):
    try:
        # Checks For Invalid User Input
        if user_id <= 0:
            st.warning("⚠️ Please enter a valid User ID (greater than 0).")
        elif book_id not in books_df["id"].values:
            st.warning("⚠️ Selected book not found in the database.")
        else:
            # Calling hybrid_recommend Function
            recs = hybrid_recommend(user_id=user_id, book_id=book_id, top_n=top_n)

            # Checking If User Is Cold-start
            if user_id not in user_item_matrix.index:
                st.info("ℹ️ Cold-start user detected: showing content-based recommendations only.")

            if recs.empty:
                st.warning("⚠️ No recommendations found. Try another User ID or Book.")
            else:
                st.success(f"✅ Recommendations for **{book_title}**:")

                # To Show The Recommendations As Card-style Blocks
                for _, row in recs.iterrows():
                    st.markdown(f"""
                    ### 📖 {row['title']}
                    - 👨‍💻 **Author(s):** {row['authors']}
                    - 🏷️ **Genre(s):** {row['genres']}
                    - 📝 **Description:**  
                      {row.get('description', 'No description available.')}
                    ---
                    """)
    except Exception as e:
        st.error(f"An error occurred: {e}")