# -------------------------------------------------------------------------
# AUTHOR: Sean Archer
# FILENAME: knn.py
# SPECIFICATION: This program trains a KNN model and predicts temperature from samples from weather_test.csv
# FOR: CS 5990- Assignment #4
# TIME SPENT: 4 hours
# -----------------------------------------------------------*/

from typing import List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


def extract_dt_object(weather_data_point: str) -> datetime:
    format_string = '%Y-%m-%d %H:%M:%S.%f %z'
    dt_object = datetime.strptime(weather_data_point, format_string)
    return dt_object


def create_dt_arrays(date_time_values: np.ndarray):
    years, months, hours = [], [], []
    for date_time in date_time_values:
        dt_object = extract_dt_object(date_time[0])
        years.append(dt_object.year)
        months.append(dt_object.month)
        hours.append(dt_object.hour)
    year_array = np.array(years).reshape(-1, 1)
    month_array = np.array(months).reshape(-1, 1)
    hour_array = np.array(hours).reshape(-1, 1)
    return year_array, month_array, hour_array


def prepare_dataset(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df_arr = df.to_numpy()
    date_time_values = df_arr[1:, 0:1]
    year_array, month_array, hour_array = create_dt_arrays(date_time_values)
    other_features = df_arr[1:, 1:-1:].astype('f')
    x_values = np.hstack((year_array, month_array, hour_array, other_features))
    y_values_raw = df_arr[1:, -1:].astype('f')
    y_values = np.array([classify(classes, val) for val in y_values_raw.flatten()])
    return x_values, y_values


def classify(classes: List[int], sample_value: float) -> int:
    distances = {}
    for class_value in classes:
        distances[class_value] = abs(class_value - sample_value)
    return int(min(distances, key=distances.get))


def prediction_difference(predicted_value: int, real_value: int) -> float:
    return abs(100 * ((predicted_value - real_value)) / real_value)


classes = [i for i in range(-22, 40, 6)]

k_values = [i for i in range(1, 20)]
p_values = [1, 2]
w_values = ['uniform', 'distance']

with open('weather_training.csv', 'r') as f:
    df_train = pd.read_csv(f, header=None)
    X_training, y_training = prepare_dataset(df_train)
    f.close()

with open('weather_test.csv', 'r') as f:
    df_test = pd.read_csv(f, header=None)
    X_test, y_test = prepare_dataset(df_test)
    f.close()

highest_accuracy = 0.0
for k in k_values:
    for p in p_values:
        for w in w_values:
            clf = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)
            pca = PCA(n_components=3)

            # X_training = pca.fit_transform(X_training)
            # X_test = pca.transform(X_test)

            clf.fit(X_training, y_training)
            correct_predictions = 0
            for (x_test_sample, y_test_sample) in zip(X_test, y_test):
                y_pred = clf.predict([x_test_sample])[0]
                diff = prediction_difference(y_pred, y_test_sample)
                if diff <= 15:
                    correct_predictions += 1
            accuracy = correct_predictions / len(y_test)
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                print(f"Highest KNN accuracy so far: {highest_accuracy}\nParameters: k={k}, p={p}, w={w}")
