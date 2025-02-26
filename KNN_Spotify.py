import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path, encoding='latin1')
    
    # Convert 'streams' and chart/playlists columns to numeric, coercing errors to NaN
    numeric_columns = ['streams', 'in_spotify_charts', 'in_apple_charts', 'in_deezer_charts', 'in_shazam_charts', 
                       'in_spotify_playlists', 'in_apple_playlists', 'in_deezer_playlists']
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    
    # Drop rows with NaN in these crucial columns
    df.dropna(subset=numeric_columns, inplace=True)
    
    # Create total charts and total playlists features
    df['total_charts'] = df['in_spotify_charts'] + df['in_apple_charts'] + df['in_deezer_charts'] + df['in_shazam_charts']
    df['total_playlists'] = df['in_spotify_playlists'] + df['in_apple_playlists'] + df['in_deezer_playlists']
    
    # Define 'hit' based on multiple criteria
    df['is_hit'] = ((df['streams'] >= df['streams'].quantile(0.9)) |
                    ((df['total_charts'] >= df['total_charts'].quantile(0.90)) &
                     (df['total_playlists'] >= df['total_playlists'].quantile(0.90)))).astype(int)
    
    # Select features for the model
    features = df[['artist_count', 'released_year', 'released_month', 'released_day',
                   'in_spotify_playlists', 'in_spotify_charts', 'in_apple_playlists',
                   'danceability_%', 'valence_%', 'energy_%', 'acousticness_%',
                   'instrumentalness_%', 'liveness_%', 'speechiness_%', 'key', 'mode']]
    
    # Ensure 'mode' is numerical
    if df['mode'].dtype == object:
        df['mode'] = df['mode'].map({'Major': 1, 'Minor': 0})
    
    return features, df['is_hit']

def train_and_evaluate_knn(X, y):
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.select_dtypes(include=np.number))
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Initialize the KNN classifier with 5 neighbors
    knn = KNeighborsClassifier(n_neighbors=5)
    
    # Fit the model to the training data
    knn.fit(X_train, y_train)
    
    # Predict on the testing set
    y_pred = knn.predict(X_test)
    
    # Calculate and return the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Adjust the file_path to the actual location of your dataset
file_path = 'Popular_Spotify_Songs.csv'

# Load and prepare the data
X, y = load_and_prepare_data(file_path)

# Train the KNN classifier and calculate accuracy
accuracy = train_and_evaluate_knn(X, y)

print(f'Model Accuracy: {accuracy}')

