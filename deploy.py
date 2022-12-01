import streamlit as st

import pickle
import pandas as pd
import plotly.graph_objects as go

animes_dict = pickle.load(open('animes.pkl','rb'))
animes = pd.DataFrame(animes_dict)

st.title('Anime Recommender System')

selected_movie_name = st.selectbox(
"Type or select an anime from the dropdown",
 animes['Title'].values
)

data = pd.read_csv('cleaned_anime.csv')

from sklearn.feature_extraction.text import TfidfVectorizer


tfidf = TfidfVectorizer(stop_words='english')
data['Synopsis'] = data['Synopsis'].fillna('')
tfidf_matrix = tfidf.fit_transform(data['Synopsis'])


from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(data.index, index=data['Title']).drop_duplicates()

@st.cache
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return data[['Title', 'Synopsis']].iloc[movie_indices]

def Table(df):
    fig=go.Figure(go.Table( columnorder = [1,2,3],
          columnwidth = [10,28],
            header=dict(values=[' Title','Description'],
                        line_color='black',font=dict(color='black',size= 19),height=40,
                        fill_color='#dd571c',#
                        align=['left','center']),
                cells=dict(values=[df.title,df.description],
                       fill_color='#ffdac4',line_color='grey',
                           font=dict(color='black', family="Lato", size=16),
                       align='left')))
    fig.update_layout(height=600, title ={'text': "Top 10 Anime Recommendations", 'font': {'size': 22}},title_x=0.5
                     )
    return st.plotly_chart(fig,use_container_width=True)
    

if st.button('Show Recommendation'):
      recommended_movie_names = get_recommendations(selected_movie_name)
      Table(recommended_movie_names)
      
      
