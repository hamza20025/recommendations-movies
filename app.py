import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import json

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


from sklearn.feature_extraction.text import TfidfVectorizer

netflix_data = pd.read_csv('Films.csv')


tfidf = TfidfVectorizer(stop_words='english')
netflix_data['Category'] = netflix_data['Category'].fillna('')
tfidf_matrix = tfidf.fit_transform(netflix_data['Category'])

from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(netflix_data.index, index=netflix_data['Title']).drop_duplicates()

def get_recommendations(Title, cosine_sim=cosine_sim):
    idx = indices[Title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return netflix_data.iloc[movie_indices]

#print(get_recommendations('Dark'))
#tfidf_matrix.shape
def Table(df):
    fig=go.Figure(go.Table( columnorder = [1,2,3],
          columnwidth = [10,28],
            header=dict(values=[' Title','Category'],
                        line_color='black',font=dict(color='black',size= 19),height=40,
                        fill_color='#dd571c',#
                        align=['left','center']),
                cells=dict(values=[df.Title,df.Category],
                       fill_color='#ffdac4',line_color='black',
                           font=dict(color='black', family="Lato", size=16),
                       align='left')))
    fig.update_layout(height=500, title ={'text': "Top 10 Movie Recommendations", 'font': {'size': 22}},title_x=0.5
                     )
    return st.plotly_chart(fig,use_container_width=True)
movie_list = netflix_data['Title'].values


####################################################################
#streamlit
##################################################################

st.header('Système de recommandation de film')
lottie_coding = load_lottiefile("m4.json")
st_lottie(
    lottie_coding,
    speed=1,
    reverse=False,
    loop=True,
    quality="low",height=220
)
selected_movie = st.selectbox(
    "Tapez ou sélectionnez un film dans la liste déroulante",
    movie_list
)

if st.button('Afficher la recommandation'):
    recommended_movie_names = get_recommendations(selected_movie)
    #list_of_recommended_movie = recommended_movie_names.to_list()
   # st.write(recommended_movie_names[['title', 'description']])
    Table(recommended_movie_names)
    
st.write('  '
         )
st.write(' ')

EDA = st.checkbox('Show Netflix Exploratory Data Analysis')
