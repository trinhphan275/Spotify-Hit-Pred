import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the dataset
df = pd.read_csv('/Users/itsnk/Desktop/Coding/CS412/FPtest/Popular_Spotify_Songs.csv', encoding='ISO-8859-1')
# print(df)
df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
df.dropna(subset=['streams'], inplace=True)
stream_threshold = df['streams'].quantile(0.90)  # Top 10% of streams
df['is_hit'] = np.where(df['streams'] >= stream_threshold, 1, 0)

# Selecting features for the model
# features = [
#     'artist_count', 'released_year', 'released_month', 'released_day',
#     'in_spotify_playlists', 'in_spotify_charts', 'in_apple_playlists',
#     'danceability_%', 'valence_%', 'energy_%', 'acousticness_%',
#     'instrumentalness_%', 'liveness_%', 'speechiness_%', 'key', 'mode'
# ]

# Scale the features
scaler = StandardScaler()
numeric_features = df[['danceability_%', 'energy_%', 'valence_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']]  # Include relevant features
numeric_features_scaled = scaler.fit_transform(numeric_features)

# Create a new column transformer that includes one-hot encoding for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['danceability_%', 'energy_%', 'valence_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']),
        ('cat', OneHotEncoder(), ['key', 'mode'])
    ])

# Ensure all_features is a list of column names
all_features = ['danceability_%', 'energy_%', 'valence_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%', 'key', 'mode']

song_features = df[all_features]  # This should work as before

# Now, this should work without the ValueError
processed_features = preprocessor.fit_transform(song_features)

# Compute the cosine similarity matrix using processed_features
similarity_matrix = cosine_similarity(processed_features)  # This should be processed_features

# Updated function to get the most similar songs should remain the same, as it operates on indices
def get_similar_songs(song_index, top_n=5):
    song_similarities = similarity_matrix[song_index]
    similar_indices = np.argsort(-song_similarities)[1:top_n+1]  # Exclude the song itself
    similar_songs = df.iloc[similar_indices]
    return similar_songs


# Test the function with a song index, for example, get similar songs to the first one in the dataset
print(get_similar_songs(0, top_n=10))
