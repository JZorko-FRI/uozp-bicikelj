import requests

import matplotlib.pyplot as plt
plt.rcParams.update({'figure.dpi': 300, 'font.size': 2})
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from secret import api_key

# Function that uses the google maps API to get the travel time on foot between two sets of coordinates
def get_travel_time(start_lat, start_lon, end_lat, end_lon):
    url = "https://maps.googleapis.com/maps/api/distancematrix/json?units=metric&origins=" + str(start_lat) + "," + str(start_lon) + "&destinations=" + str(end_lat) + "," + str(end_lon) + "&mode=walking&key=" + api_key
    response = requests.get(url)
    json = response.json()
    if json['status'] == 'OK':
        return float(json['rows'][0]['elements'][0]['duration']['value'])
    else:
        return None

def compute_distance_matrix(df_metadata):
    df_distances = pd.DataFrame(index=df_metadata.index, columns=df_metadata.index, dtype=float)
    for i, row in tqdm(df_metadata.iterrows()):
        for j, col in df_metadata.iterrows():
            if i <= j:
                df_distances.loc[i, j] = 0
            else:
                distance = get_travel_time(row['lat'], row['lon'], col['lat'], col['lon'])
                # distance = 1
                df_distances.loc[i, j] = distance
                df_distances.loc[j, i] = distance
    return df_distances



if __name__ == "__main__":
    df_metadata = pd.read_csv('data/bicikelj_metadata.csv', sep='\t')

    df_metadata.rename(columns={
        'postaja': 'Station Name',
        'geo-visina': 'lat',
        'geo-sirina': 'lon'
    }, inplace=True)

    df_metadata.set_index('Station Name', inplace=True)
    df_metadata.sort_index(inplace=True)
    df_metadata.drop(columns=['total_space'], inplace=True)

    df_distances = compute_distance_matrix(df_metadata)

    df_distances.to_csv('data/bicikelj_distances.csv', index=True)

    sns.heatmap(df_distances, fmt='.0f', cmap='rocket', annot=True)
    plt.show()
