import numpy as np
import pandas as pd
import streamlit as sl
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


netflix = pd.read_csv('netflix_cleaned.csv')
netflix_data = netflix.copy()
df = netflix


netflix_tfid = TfidfVectorizer(stop_words='english')
netflix_data['description'] = netflix_data['description'].fillna('')
netflix_tfidf_matrix = netflix_tfid .fit_transform(netflix_data['description'])

cosine_sim = linear_kernel(netflix_tfidf_matrix, netflix_tfidf_matrix)

indices = pd.Series(netflix_data.index, index=netflix_data['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return netflix_data[['title','description']].iloc[movie_indices]

movie_list = netflix_data['title'].values



sl.title(":red[Net Recommender]")
sl.markdown("Quick and easy way to decide on which movie to watch")

with sl.sidebar:
    choose = option_menu("MENU", ["Home","Recommend"],
                         icons=['house','funnel-fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "black"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "Red"},
        "nav-link-selected": {"background-color": "Brown"},
    }
    )

if choose == "Home":
    col1, col2 = sl.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        sl.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Helvetica'; color: Orange;}
         </style> """, unsafe_allow_html=True)
        sl.markdown('<p class="font">Objective</p>', unsafe_allow_html=True)    
        sl.write("The primary goal of a Netflix show recommender is to offer personalized recommendations to users based on their viewing history, ratings, genre preferences, and other relevant data. This is achieved through the utilization of advanced machine learning techniques and algorithms that analyze user behavior and patterns to predict what type of content a user might enjoy.")
        sl.write("One common technique used is collaborative filtering, where the system compares the user's preferences with those of other users and identifies individuals with similar tastes. The system then recommends content that these similar users have enjoyed in the past. Alternatively, content-based filtering involves recommending shows or movies that share similar attributes with those the user has previously shown interest in, like genres, actors, directors, and themes.")

  


if choose == "Recommend":
    col1, col2 = sl.columns( [0.9, 0.1])
    with col1:               # To display the header text using css style
        sl.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Helvetica'; color: Red;} 
        </style> """, unsafe_allow_html=True)
        selected_movie = sl.selectbox( "Type or Select your movie ", movie_list )
        if sl.button('Recommend'):
            recommended_movie_names = get_recommendations(selected_movie)
            recommended_movie_names
