# Importation de librairie
import numpy as np
import pandas as pd
import streamlit as st

# lecture du fichier 
df = pd.read_csv('Films.csv')
df.head()

#we are need only two column form above dataset 
df[['Title','Category']].head()

# Libraire d'extraction de texte
from sklearn.feature_extraction.text import TfidfVectorizer

# Copie du Dataframe
netflix_data = df.copy()

# Extraction de text 
tfidf = TfidfVectorizer(stop_words='english')
netflix_data['Category'] = netflix_data['Category'].fillna('')
tfidf_matrix = tfidf.fit_transform(netflix_data['Category'])


from sklearn.feature_extraction.text import TfidfVectorizer

# Calcul de la Matrice 
from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(netflix_data.index, index=netflix_data['Title']).drop_duplicates()

# Recommandation
def get_recommendations(Title, cosine_sim=cosine_sim):
    idx = indices[Title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return netflix_data[['Title','Category']].iloc[movie_indices]

# Afficher la commande 
get_recommendations('Top Gun: Maverick')


# Titre
st.title('Système de recommandation de film')

st.subheader('Application réalisée par Hermann')

st.markdown('Cette application permet de recommander des films')

# Pour insérer les images
col1, col2, col3, col4, col5 = st.columns(5)
   
with col2:
    st.header('Drame')
    st.image("Joker_drame.jpg")

    
with col3:
    st.header("Thriller")
    st.image("Crash1996.jpg")

    
with col4:
    st.header("Fiction")
    st.image("Stars8.jpg")

with col5:
    st.header("Comédie")
    st.image("Capitain_comedy.jpg")

with col1: 
    st.header("Action")
    st.image('top_gun_3.jpg')

    
df = pd.read_csv('Films.csv')
df1 = df['Title'].values

selected_movie =st.selectbox("Tapez ou sélectionnez un film dans la liste déroulante", df1)


if st.button('Afficher la recommandation'):
      recommended_movie_names = get_recommendations(selected_movie)
      recommended_movie_names 
