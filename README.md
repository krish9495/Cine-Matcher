# Movie Recommender System

A content-based movie recommendation system built with Python and Streamlit.

## Setup Instructions

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook `Movie-recommeder-system.ipynb` to generate the pickle files:
   - This will create `similarity.pkl`, `movie_dict.pkl`, and `movies.pkl`
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Features

- Content-based movie recommendations
- Movie poster display
- Clean and intuitive user interface
- Top 5 similar movie recommendations

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- TMDB API 