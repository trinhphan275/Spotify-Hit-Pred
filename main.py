import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

# Load the dataset
df = pd.read_csv('/Users/itsnk/Desktop/Coding/CS412/FPtest/Popular_Spotify_Songs.csv', encoding='ISO-8859-1')
# print(df.columns)
df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
df.dropna(subset=['streams'], inplace=True)

chart_columns = ['in_spotify_charts', 'in_apple_charts', 'in_deezer_charts', 'in_shazam_charts']
for col in chart_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert errors to NaN

playlist_columns = ['in_spotify_playlists', 'in_apple_playlists', 'in_deezer_playlists']
for col in playlist_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert errors to NaN

# Fill NaN values with 0 (if that's appropriate for your analysis)
df[chart_columns] = df[chart_columns].fillna(0)

# Fill NaN values with 0 (if that's appropriate for your analysis)
df[playlist_columns] = df[playlist_columns].fillna(0)

# Feature Engineering
# Combine different metrics to create composite popularity scores
df['total_charts'] = df['in_spotify_charts'] + df['in_apple_charts'] + df['in_deezer_charts'] + df['in_shazam_charts']
df['total_playlists'] = df['in_spotify_playlists'] + df['in_apple_playlists'] + df['in_deezer_playlists']

# Define 'hit' based on multiple criteria
# Here I'm considering a song a 'hit' if it's high on total charts and total playlists, you can adjust the thresholds
df['is_hit'] = ((df['streams'] >= df['streams'].quantile(0.9)) |
                ((df['total_charts'] >= df['total_charts'].quantile(0.90)) &
                 (df['total_playlists'] >= df['total_playlists'].quantile(0.90)))).astype(int)

# Selecting features for the model
features = [
    'artist_count', 'released_year', 'released_month', 'released_day',
    'in_spotify_playlists', 'in_spotify_charts', 'in_apple_playlists',
    'danceability_%', 'valence_%', 'energy_%', 'acousticness_%',
    'instrumentalness_%', 'liveness_%', 'speechiness_%', 'key', 'mode'
]
X = df[features]
y = df['is_hit']

# Split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Setting up preprocessing for numeric and categorical features
numeric_features = [col for col in features if col not in ['key', 'mode']]  # All numeric features
categorical_features = ['key', 'mode']  # All categorical features

# Create a preprocessing and modeling pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=5000),
    'Random Forest': RandomForestClassifier(random_state=10000, bootstrap=True, class_weight='balanced'),
    'Neural Network': MLPClassifier(max_iter=1000, random_state=1000, alpha=0.1)
}

# Set up 10-fold cross-validation
cv = StratifiedKFold(n_splits=10)

# Perform cross-validation and print classification report for each model
for name, model in models.items():
    print(f"{name} Results:")
    y_true, y_pred_list = [], []

    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        pipeline = make_pipeline(preprocessor, model)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        y_true.extend(y_test)
        y_pred_list.extend(y_pred)

    print(classification_report(y_true, y_pred_list))
    print('-----------------------------------------------------\n')

# pipeline = make_pipeline(preprocessor, LogisticRegression(max_iter=1000))
#
# # Train the logistic regression model on the training data
# pipeline.fit(X_train, y_train)
#
# # Make predictions and evaluate the model
# y_pred = pipeline.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))


################################################################################################

# File path
# file_path = '/Users/itsnk/Desktop/Coding/CS412/FPtest/spotify-2023.csv'
#
# # Initialize an empty list to store the column names (features)
# features = []
#
# # Open the CSV file and extract the column names
# with open(file_path, mode='r', encoding='ISO-8859-1') as file:
#     # Create a CSV reader object
#     csv_reader = csv.reader(file)
#
#     # Extract the first row (column names)
#     for row in csv_reader:
#         features = row
#         break  # Only read the first row
#
# # Display the extracted features
# print(features)

