from csv import writer

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def run(feature_size=50,
        target=5,
        bar=0.0,
        epochs=40,
        batch_size=1000,
        val_split=0.2,
        prec_thresholds=0.95,
        folder_path="data/20100101-20200101/",
        save_model=False):
    model_name = "m_" + str(feature_size) + "_" + str(target) + "_" + str(bar) + "_" + \
                 folder_path.split("/")[1]
    x, x_test, y, y_test = get_data(
        folder_path + "dataset/" + str(feature_size) + "_" + str(target) + "/", bar)
    my_model = train(feature_size, x, y, batch_size, epochs, val_split, prec_thresholds)
    if save_model:
        my_model.save(model_name)
    my_model.evaluate(x=x_test, y=y_test)

def get_data(pth, bar=0.0):
    print('\r', "Reading data... ", end='')
    x = pd.read_csv(pth + "x.csv")
    y = pd.read_csv(pth + "y.csv")["Delta"]
    for i in range(len(y)):
        if y[i] > bar:
            y[i] = 1.0
        else:
            y[i] = 0.0
    x, y = shuffle(x, y)
    x, x_test, y, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    print("Done. ")
    return x, x_test, y, y_test


def train(feature_size, x, y, batch_size, epochs, val_split, prec_thresholds):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(feature_size,)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(thresholds=prec_thresholds)])
    model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=val_split, verbose=2)
    return model


def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)
