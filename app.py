import streamlit as st
import pickle
import pandas as pd
import requests
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import nltk
from nltk.stem.porter import PorterStemmer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    recommended_movie_posters = []
    
    for i in movies_list:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movie_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_movie_posters

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

def stem(text):
    ps = PorterStemmer()
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

def generate_pickle_files():
    # Load the datasets
    movies = pd.read_csv('Datasets/tmdb_5000_movies.csv')
    credits = pd.read_csv('Datasets/tmdb_5000_credits.csv')
    
    # Merge the datasets
    movies = movies.merge(credits, on='title')
    
    # Select relevant columns
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    
    # Drop rows with missing values
    movies.dropna(inplace=True)
    
    # Convert string representations to lists
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    
    # Convert overview to list of words
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    
    # Remove spaces from genres, keywords, cast, and crew
    movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
    
    # Create tags column
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    
    # Convert tags to string
    movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))
    
    # Convert to lowercase
    movies['tags'] = movies['tags'].apply(lambda x: x.lower())
    
    # Apply stemming
    movies['tags'] = movies['tags'].apply(stem)
    
    # Create new dataframe with only required columns
    new_df = movies[['movie_id', 'title', 'tags']]
    
    # Create vectors
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    
    # Calculate similarity
    similarity = cosine_similarity(vectors)
    
    # Save the pickle files
    pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))
    pickle.dump(similarity, open('similarity.pkl', 'wb'))

# Set up the Streamlit page
st.set_page_config(page_title="Movie Recommender System", layout="wide")
st.title('Movie Recommender System')

# Check if pickle files exist
if not (os.path.exists('movie_dict.pkl') and os.path.exists('similarity.pkl')):
    st.warning("""
    Required pickle files are missing. You can either:
    1. Upload existing pickle files
    2. Generate new pickle files from the datasets
    """)
    
    # File upload option
    uploaded_files = st.file_uploader("Upload pickle files", type=['pkl'], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            with open(file.name, 'wb') as f:
                f.write(file.getbuffer())
        st.success("Files uploaded successfully!")
    
    # Generate files option
    if st.button('Generate Pickle Files'):
        with st.spinner('Generating pickle files... This may take a few minutes.'):
            try:
                generate_pickle_files()
                st.success("Pickle files generated successfully!")
            except Exception as e:
                st.error(f"Error generating pickle files: {str(e)}")
else:
    # Load data
    movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
    similarity = pickle.load(open('similarity.pkl', 'rb'))

    # Create a dropdown to select a movie
    selected_movie = st.selectbox(
        'Select a movie you like',
        movies['title'].values
    )

    # Add a button to get recommendations
    if st.button('Get Recommendations'):
        names, posters = recommend(selected_movie)
        
        # Display the recommendations in 5 columns
        cols = st.columns(5)
        
        for idx, (col, name, poster) in enumerate(zip(cols, names, posters)):
            with col:
                st.text(name)
                st.image(poster) 