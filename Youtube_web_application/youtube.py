

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from streamlit_option_menu import option_menu
from streamlit_multi_menu import streamlit_multi_menu


# Set wide layout mode
st.set_page_config(layout="wide")

# Load the YouTube video data (replace with your own data)
df = pd.read_pickle('Pickle_file.gz')

# Replace NaN and inf values
df[['likeCount', 'viewCount']] = df[['likeCount', 'viewCount']].fillna(0)  # Replace NaN with 0
df[['likeCount', 'viewCount']] = df[['likeCount', 'viewCount']].replace([np.inf, -np.inf], 0)
df['likeCount'] = df['likeCount'].astype(int)
df['viewCount'] = df['viewCount'].astype(int)
# Ensure that all tags are strings and replace NaN/None with an empty string
df['tags'] = df['tags'].fillna('').astype(str)
# Remove rows where 'tags' is still an empty string after preprocessing
df = df[df['tags'].str.strip() != '']

def home_page():
    # Display the YouTube icon image
    st.image("https://vectorseek.com/wp-content/uploads/2021/01/youtube-logo-download.png",width=100)
    st.title('Welcome to YouTube Video Recommender')

    # Introduction
    st.write("""
        This application helps you discover YouTube videos based on your search query or chosen category.
        You can search for specific videos using the search box or explore videos by selecting a category.
    """)

    # Instructions
    st.subheader('How to Use:')
    st.write("""
        - **Search:** Type a video title or keyword in the search box on the "Search" page and get recommendations.
        - **Categories:** Select a category from the "Category" page to explore videos in that genre.
    """)

    # Highlight Popular Categories
    st.subheader('Popular Categories:')
    st.write("""
        - News
        - Sports
        - Songs & Music
        - Education
        - Food
        - Comedy & Entertainment
        - Travel
        - Tech
        - Fashion
        - Lifestyle
        - Wildlife
        - Economy
        - Finance
    """)
    st.subheader('About My Contribution:')
    st.write("""
        I am Anafa Julaiha, and I developed this YouTube video recommender system as part of my project work. 
        My role involved data collection, preprocessing, feature engineering, and the implementation of machine learning models 
        for recommending videos based on user input. I also designed and developed the web interface using Streamlit to 
        provide an intuitive and interactive user experience.
    """)


def search():
    st.image("https://wallpapers.com/images/hd/youtube-logo-2272-x-1260-background-ybrih3rlwxycir54.jpg",width=200)
    st.title("Search")
    search_input = st.text_input("Search for videos")
    search_button = st.button("Search")
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tag_vectors = vectorizer.fit_transform(df['tags'])

    def recommend_videos(user_input, df, tag_vectors, vectorizer, top_n=500):
        user_vector = vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vector, tag_vectors).flatten()
        top_indices = similarities.argsort()[-top_n:][::-1]
        recommendations = df.iloc[top_indices][['video_id', 'channelTitle', 'title', 'thumbnail_url','likeCount','viewCount']]
        #unique_recommendations = recommendations.drop_duplicates(subset='channelTitle')
        return recommendations

    if search_button and search_input:
        recommendations = recommend_videos(search_input, df, tag_vectors, vectorizer)
        if recommendations.empty :  # Check if no recommendations were found
            st.subheader("No videos found related to your search.")
        else:
            for index, row in recommendations.iterrows():
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f'<a href="https://www.youtube.com/watch?v={row["video_id"]}" target="_blank">'
                                f'<img src="{row["thumbnail_url"]}" style="width: 100%"></a>', 
                                unsafe_allow_html=True)
                with col2:
                    st.markdown(f"**{row['title']}**")
                    st.text(f"Channel: {row['channelTitle']}")
                    st.text(f"Views: {row['viewCount']}")
                    st.text(f"Likes: {row['likeCount']}")


def category():
    st.image("https://vectorseek.com/wp-content/uploads/2021/01/youtube-logo-download.png",width=100)
    sub_menus ={"GENERAL": ['News', 'Economy & Finance', 'Education', 'Tech' ],"ENTERTAINMENT" :['Songs & Music','Sports','Comedy','Wildlife'],"OTHERS":['Food','Fashion','Lifestyle','Travel']}
    selected_menu = streamlit_multi_menu(menu_titles=list(sub_menus.keys()),
                              sub_menus=sub_menus,sub_menu_text_align = 'center',
                              sub_menu_color = "#990000",
                              sub_menu_font_color =  "#FFFFFF",
                            use_container_width=True)
    
    def custom_preprocessor(doc):
        if doc is None:
            return ""
        return doc.lower()
    # Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english',preprocessor=custom_preprocessor)

    # Fit and transform the tags data
    tag_vectors = vectorizer.fit_transform(df['tags'])
    

    def recommend_category_videos(category, df, tag_vectors, vectorizer):
        category_vector = vectorizer.transform([category])
        similarities = cosine_similarity(category_vector, tag_vectors).flatten()
        top_indices = similarities.argsort()[-50:][::-1]
        recommendations = df.iloc[top_indices][['video_id', 'channelTitle', 'title', 'thumbnail_url','likeCount','viewCount']]
        #unique_recommendations = recommendations.drop_duplicates(subset='channelTitle')
        return recommendations

    if selected_menu != 'All':
        recommendations = recommend_category_videos(selected_menu, df, tag_vectors, vectorizer)

        if recommendations.empty:
            st.write("No videos found for this category.")
        else:
            for index, row in recommendations.iterrows():
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f'<a href="https://www.youtube.com/watch?v={row["video_id"]}" target="_blank">'
                                f'<img src="{row["thumbnail_url"]}" style="width: 100%"></a>', 
                                unsafe_allow_html=True)
                with col2:
                    st.markdown(f"**{row['title']}**")
                    st.text(f"Channel: {row['channelTitle']}")
                    st.text(f"Views: {row['viewCount']}")
                    st.text(f"Likes: {row['likeCount']}")
    else:
        st.write("Please select a category.") 
    def _preprocess(doc):
        if doc is None:
            return ""
        return doc.lower()

with st.sidebar: 
    selected = option_menu(
        menu_title="Main Menu", 
        options=["Home", "Search", "Category"],
        icons=["house", "search", "list"],  # Add appropriate icons
        menu_icon="cast",
        default_index=0,
    )

# Display the selected page
if selected == "Search":
    search()
elif selected == "Home":
    home_page()
elif selected == "Category":
    category()



