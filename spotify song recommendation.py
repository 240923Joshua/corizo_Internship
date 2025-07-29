import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

df = pd.read_csv(r"C:\Users\PROBOOK\Downloads\spotify.csv")

df = df.drop(columns=[
    'track_id', 'track_album_id', 'track_album_name',
    'track_album_release_date', 'playlist_id', 'track_artist'
], errors='ignore')

df = df.dropna()

df['playlist_genre'] = df['playlist_genre'].astype('category').cat.codes
df['playlist_name'] = df['playlist_name'].astype('category').cat.codes
df['playlist_subgenre'] = df['playlist_subgenre'].astype('category').cat.codes


audio_features = ['danceability','energy','loudness','speechiness','acousticness',
                  'instrumentalness','liveness','valence','tempo','duration_ms']
df[audio_features].hist(bins=20, figsize=(12,8), color='skyblue', edgecolor='black')
plt.suptitle("Histograms of Audio Features")
plt.tight_layout()
plt.show()

corr = df[['track_popularity'] + audio_features].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Numeric Features")
plt.show()

features = ['track_popularity'] + audio_features + ['playlist_genre', 'playlist_name', 'playlist_subgenre']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)
df['pca1'] = pca_components[:, 0]
df['pca2'] = pca_components[:, 1]


kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)


plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='Set1', alpha=0.7)
plt.title('K-Means Clustering Visualization using PCA', fontsize=16)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

def recommend_song(track_name):
    if track_name not in df['track_name'].values:
        print(f"Track '{track_name}' not found in the dataset.")
        return

    song = df[df['track_name'] == track_name].iloc[0]
    cluster = song['cluster']
    genre = song['playlist_subgenre']
    genreG = song['playlist_genre']
    mode = song['mode']
    song['energy'] *= 10
    df['energy'] *= 10
    energy = song['energy'].astype(int)
    energydf = df['energy'].astype(int)
    song['tempo'] //=10
    df['tempo'] //= 10
    tempo = song['tempo'].astype(int)
    tempodf = df['tempo'].astype(int)
    song['liveness'] *= 10
    df['liveness'] *= 10
    live = song['liveness'].astype(int)
    livedf = df['liveness'].astype(int)
    
    recommendations = df[
        (df['cluster'] == cluster) &
        (df['playlist_genre'] == genreG) &
        (df['playlist_subgenre'] == genre)& 
        (df['mode'] == mode)
        &((energy == energydf) | (tempo == tempodf) | (live == livedf))
        &(df['track_name'] != track_name)
    ]['track_name'].drop_duplicates().head(3)

    print(f"\nRecommendations for '{track_name}':")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

song = input("Enter a song : ")
recommend_song(song)

