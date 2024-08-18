# YouTube Video Recommendation System

This project is a YouTube video recommendation system that mimics the recommendation features of YouTube by utilizing machine learning and web technologies. The system allows users to search for videos and receive recommendations across various categories like news, sports, education, and more.

## Project Overview

### Data Collection
- **YouTube API**: Collected video metadata including channel ID, playlist ID, video ID, descriptions, thumbnails, tags, view count, and like count using the YouTube Data API.
- **Data Format**: The collected data was stored in JSON format.

### Cloud Storage
- **AWS S3**: The JSON data was stored in an S3 bucket for scalable storage.
- **Database Storage**: Cleaned and structured data was stored in AWS RDS using MySQL.

### Data Processing
- **Data Cleaning**: The data was cleaned and structured using pandas to remove inconsistencies and prepare it for analysis.
- **Data Structuring**: Converted the cleaned data into a structured format and stored it in an AWS RDS instance using MySQL, SQLAlchemy, and MySQL Connector.

### Machine Learning
- **Clustering**: Used k-means clustering to group similar videos based on their features.
- **Vectorization**: Applied TF-IDF vectorization to encode textual data for machine learning.
- **Model Evaluation**: Evaluated the clustering model using the silhouette score.
- **Model Storage**: The final model was stored in pickle format and compressed using gzip.

### Web Application
- **Streamlit**: Developed a web application using Streamlit to provide a user-friendly interface for searching and viewing video recommendations.
- **UI Components**: Used `streamlit_option_menu` and `streamlit_multi_menu` for navigation and category selection.
- **Recommendation System**: The application provides video recommendations based on user input, similar to YouTube.

### Deployment
- **AWS Cloud**: The web application was deployed on an AWS Ubuntu server.
- **Access**: You can access the application [here](http://3.109.153.63:8501/).

## Categories
The recommendation system covers the following categories:
- News
- Sports
- Education
- Food
- Fashion
- Lifestyle
- Economy and Finance
- Comedy
- Wildlife
- Tech
- Travel
- Songs


