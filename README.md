# 📚  Novel Recommendation System

A project that implements a **Hybrid Recommendation Engine** for novels using **Content-Based Filtering + Collaborative Filtering**.  
Built with **Python, SQLite, Scikit-learn, and Streamlit**.

---

## ✨ Features
- Hybrid recommender system combining:
  - ✅ **Content-Based Filtering** (TF-IDF + Cosine Similarity on titles, authors, genres, descriptions)
  - ✅ **Collaborative Filtering** (SVD matrix factorization on user ratings)
- Weighted hybrid approach with adjustable parameter `alpha`
- SQLite database integration (`book_details`, `user_interactions`)
- Streamlit interface for interactive recommendations
- Error handling for cold-start problems and invalid user inputs

---

## 🛠 Tech Stack
- **Python 3.10+**
- **SQLite3** (lightweight DB)
- **Pandas, NumPy**
- **Scikit-learn** (TF-IDF, SVD, similarity, scaling)
- **Streamlit** (UI)

## ⚡ Installation & Usage

### 1️ Clone the Repository
- git clone https://github.com/VedantMPatil10/Book-Recommendation-System.git
- cd Book-Recommendation-System

### 2️⃣ Create Virtual Environment
- python -m venv .venv
- source .venv/bin/activate   # Linux/Mac
- .venv\Scripts\activate      # Windows

### 3️⃣ Install Dependencies
pip install -r requirements.txt

### 4️⃣ Run the Streamlit App
streamlit run app.py

## 🎯 Example Demo
### User enters their User ID & Selects a Book Title
<img width="1366" height="641" alt="Screenshot 2025-09-27 141813" src="https://github.com/user-attachments/assets/0c0e4be8-855f-4121-931a-b704fff80139" />

### Gets top-N personalized recommendations 🎉 
<img width="1366" height="634" alt="Screenshot 2025-09-27 142039" src="https://github.com/user-attachments/assets/ed153dae-b8fe-4e9e-bee7-a3ad2c343335" />

---

<img width="1366" height="643" alt="Screenshot 2025-09-27 142121" src="https://github.com/user-attachments/assets/efa48263-7b26-4b56-ae0c-51d3b0d3235d" />

## 📊 Future Improvements

- CLI version for quick testing.
- Deployment on free cloud (Streamlit Community Cloud / Hugging Face Spaces).
- Improved evaluation with metrics (RMSE, Precision@k, Recall@k).
---
## 👨‍💻 Author: Vedant Patil
