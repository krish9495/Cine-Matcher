import streamlit as st
import pickle
import pandas as pd
import requests

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

# Load data
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Set up the Streamlit page
st.set_page_config(page_title="Movie Recommender System", layout="wide")
st.title('Movie Recommender System')

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