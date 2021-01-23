import os
import pickle
import random
from collections import deque
from csv import writer
from datetime import datetime

import pandas as pd
import tensorflow as tf

from data_processing import get_or_create_path, normalize
from train import append_list_as_row


def generate_test_dict(model_name, x_y_folder_path):
    x, y, y_symbol, y_buy_date, y_sell_date = get_test_data(x_y_folder_path)
    generate_test_dict_given_test_files(model_name, x, y, y_symbol, y_buy_date, y_sell_date)


def generate_test_dict_given_test_files(model_name, x_test, y_test, y_symbol, y_buy_date,
                                        y_sell_date):
    dict_path = "data/" + model_name.split("_")[-1] + "/statistics/d_selected" + str(
        model_name.split("_")[-2]) + ".pkl"
    model = tf.keras.models.load_model(model_name)
    predictions = model.predict(x_test)
    d_selected = {}
    print('\r', "Generating test dict... ", end='')
    for i in range(len(predictions)):
        buy_date, sell_date = y_buy_date[i], y_sell_date[i]
        delta = y_test[i]
        sym = y_symbol[i]
        d_selected[(buy_date, sell_date, sym)] = (delta, predictions[i][0])
    print("Done. ")
    pickle.dump(d_selected, open(dict_path, "wb"))


def get_test_data(pth):
    print('\r', "Reading data... ", end='')
    x = pd.read_csv(pth + "x.csv")
    y = pd.read_csv(pth + "y.csv")["Delta"]
    y_symbol = pd.read_csv(pth + "y.csv")["Symbol"]
    y_buy_date = pd.read_csv(pth + "y.csv")["Buy date"]
    y_sell_date = pd.read_csv(pth + "y.csv")["Sell date"]
    print("Done. ")
    return x, y, y_symbol, y_buy_date, y_sell_date


def generate_dataset_for_test(feature_size, target, path, ref_path="data/20100101-20200101/"):
    input_path = get_or_create_path(path + "processed/")
    output_path = get_or_create_path(
        path + "dataset/" + str(feature_size) + '_' + str(target) + '/')
    # Delete files
    filenames = ["x.csv", "y.csv"]
    for filename in filenames:
        if os.path.isfile(output_path + filename):
            os.remove(output_path + filename)
    arr_processed = os.listdir(input_path)
    mu, std = pickle.load(open(ref_path + "statistics/mu_std.pkl", "rb"))
    with open(output_path + "x.csv", "a+", newline='') as x_csv, open(
            output_path + "y.csv", "a+", newline='') as y_csv:
        x_writer = writer(x_csv)
        y_writer = writer(y_csv)
        x_writer.writerow([i for i in range(feature_size)])
        y_writer.writerow(["Symbol", "Buy date", "Sell date", "Delta"])
        for idx, filename in enumerate(arr_processed):
            print('\r', "Producing dataset from " + str(idx + 1) + '/' + str(
                len(arr_processed)) + " stocks... ", end='')
            df = pd.read_csv(input_path + filename)
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)
            length = len(df)
            sym = filename.split(".")[0]
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


def evaluate_models(model_name_list, thresholds):
    model_dicts = []
    tp = total_cnt = total_delta = 0
    d_final = {}
    for n in model_name_list:
        d_path = "data/" + n.split("_")[-1] + "/statistics/d_selected" + n.split("_")[-2] + ".pkl"
        model_dicts.append(pickle.load(open(d_path, "rb")))
    for key in sorted(model_dicts[0].keys(),
                      key=lambda date: datetime.strptime(date[0], '%Y-%m-%d')):
        if all(cur_dict[key][1] > thresholds for cur_dict in model_dicts):
            d_final[key] = [model_dicts[0][key][0], [cur_dict[key][1] for cur_dict in model_dicts]]

    for (buy_date, sell_date, sym) in sorted(d_final.keys(),
                                             key=lambda date: datetime.strptime(date[0],
                                                                                '%Y-%m-%d')):
        year, month, day = sell_date.split("-")
        year, month, day = int(year), int(month), int(day)

        if year:
            total_cnt += 1
            delta, predictions = d_final[(buy_date, sell_date, sym)]
            total_delta += delta

            if delta > 0:
                tp += 1
            # print(buy_date, sell_date, sym, delta)

    def simulate():
        budget = 10000
        prev_sell_date = datetime.strptime("1000-01-01", '%Y-%m-%d')
        for (buy_date, sell_date, sym) in sorted(d_final.keys(),
                                                 key=lambda date: datetime.strptime(date[0],
                                                                                    '%Y-%m-%d')):
            cur_sell_date = datetime.strptime(sell_date, '%Y-%m-%d')
            cur_buy_date = datetime.strptime(buy_date, '%Y-%m-%d')
            delta, predictions = d_final[(buy_date, sell_date, sym)]

            if prev_sell_date < cur_buy_date:
                if random.uniform(0, 1) > 0.1:
                    prev_sell_date = cur_sell_date
                    budget -= 10
                    budget *= (1 + delta)
                    budget -= 10

                    print(buy_date, sell_date, sym, delta)
                    print(budget)

    prec = tp / total_cnt if total_cnt > 0 else 0
    lst = [model_name_list, thresholds, prec, tp, total_cnt, total_delta / total_cnt]
    append_list_as_row("logs/record.csv", lst)
    print(lst)


evaluate_models(["m_50_5_0.03_20100101-20190101", "m_50_5_0.03_20100101-20190101",
                 "m_50_5_0.03_20100101-20190101"], 0.9)
