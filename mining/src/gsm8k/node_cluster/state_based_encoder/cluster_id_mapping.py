import numpy as np
from sklearn.cluster import KMeans

def cluster_mapping(df, n_clusters=10):
    df['cluster_id'] = None
    mask = df['similarity'].apply(lambda x: x not in ["root"])
    features = np.array([list(map(float, sim.split('_'))) for sim in df.loc[mask, 'similarity']])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df.loc[mask, 'cluster_id'] = kmeans.fit_predict(features)
    df.loc[df['similarity'] == "root", 'cluster_id'] = -1
    return df
