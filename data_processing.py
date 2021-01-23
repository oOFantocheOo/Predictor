import os
import pickle
from collections import deque
from csv import writer

import pandas as pd


def process(path):
    # Read the csv file
    df = pd.read_csv(path)
    percent_change = [0.0]
    for i in range(len(df['Open']) - 1):
        percent_change.append(
            (df['Close'][i + 1] - df['Close'][i]) / df['Close'][
                i] * 100)
    df['Delta'] = percent_change
    # Drop "Open","High","Low" columns to improve performance
    df.pop('Open')
    df.pop('High')
    df.pop('Low')
    df.pop('Adj Close')
    df.pop('Volume')
    return df


def process_all_and_save(folder_path):
    file_path = folder_path + "raw/"
    arr = os.listdir(file_path)
    save_path = get_or_create_path(folder_path + "processed/")
    for i, f in enumerate(arr):
        print('\r', "Processing " + str(i + 1) + '/' + str(len(arr)) + " files... ", end='')
        df = process(file_path + f)
        if not os.path.isfile(save_path + f):
            df.to_csv(save_path + f)
        else:
            print("file " + str(f) + " exists!")
            return
    print("Done. ")


def generate_dataset(feature_size, target, path):
    input_path = get_or_create_path(path + "processed/")
    output_path = get_or_create_path(
        path + "dataset/" + str(feature_size) + '_' + str(target) + '/')
    # Delete files
    filenames = ["x.csv", "y.csv"]
    for filename in filenames:
        if os.path.isfile(output_path + filename):
            os.remove(output_path + filename)
    arr_processed = os.listdir(input_path)

    def get_mu_std(read_prev=False):
        if not read_prev:
            arr_percent = []
            for idx, filename in enumerate(arr_processed):
                print('\r', "Collecting deltas from " + str(idx + 1) + '/' + str(
                    len(arr_processed)) + " stocks... ", end='')
                df = pd.read_csv(input_path + filename)
                df = df.dropna()
                tmp = df["Delta"].values.tolist()
                for i in range(len(tmp))[::-1]:
                    if tmp[i] >= 80:
                        tmp.pop(i)
                arr_percent.extend(tmp)
            print('\r', "Calculating mean and stdev... ", end='')
            df_delta = pd.DataFrame({"Delta": arr_percent})
            mu = float(df_delta.mean()["Delta"])
            std = float(df_delta.std()["Delta"])
            get_or_create_path(path + "statistics/")
            pickle.dump((mu, std), open(path + "statistics/mu_std.pkl", "wb"))
            print("Done. ")
            print("Mean: " + str(mu))
            print("Stdev: " + str(std))
        else:
            mu, std = pickle.load(open(path + "statistics/mu_std.pkl", "rb"))
        return mu, std

    mu, std = get_mu_std()
    with open(output_path + "x.csv", "a+", newline='') as x_csv, open(
            output_path + "y.csv", "a+", newline='') as y_csv:
        x_writer = writer(x_csv)
        y_writer = writer(y_csv)
        x_writer.writerow([i for i in range(feature_size)])
        y_writer.writerow(["Symbol", "Buy date", "Sell date", "Delta"])
        sym = filename.split(".")[0]
        for idx, filename in enumerate(arr_processed):
            print('\r', "Producing dataset from " + str(idx + 1) + '/' + str(
                len(arr_processed)) + " stocks... ", end='')
            df = pd.read_csv(input_path + filename)
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)
            length = len(df)
            if length > feature_size + target + 1:
                cur_features = deque()
                for i in range(feature_size):
                    cur_features.append(normalize(mu, std, df["Delta"][i]))
                for i in range(length - feature_size - target):
                    cur_features.popleft()
                    cur_features.append(normalize(mu, std, df["Delta"][feature_size + i]))
                    if all(df["Close"][feature_size + i - j] >= 2 for j in range(feature_size)):
                        price_buy = df["Close"][feature_size + i]
                        price_sell = df["Close"][feature_size + i + target]
                        delta = (price_sell - price_buy) / price_buy
                        buy_date = df["Date"][feature_size + i]
                        sell_date = df["Date"][feature_size + i + target]
                        x_writer.writerow(cur_features)
                        y_writer.writerow([sym, buy_date, sell_date, delta])
    print("Done. ")


def get_or_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def normalize(mean, stdev, val):
    return (val - mean) / stdev


def check(output_path):
    tmp_file_x = output_path + "x.csv"
    tmp_file_y = output_path + "y.csv"
    df_x = pd.read_csv(tmp_file_x)
    df_y = pd.read_csv(tmp_file_y)
    print(df_x.shape, df_y.shape)

