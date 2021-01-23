import os
import shutil
import time

from selenium import webdriver

date_to_second = {"20100101-20200101": (1262304000, 1577836800),
                  "20150101-20200101": (1420070400, 1577836800),
                  "20190701-20200701": (1561939200, 1593561600),
                  "20200701-20200906": (1593561600, 1599350400),
                  "20200101-20200906": (1577836800, 1599350400),
                  "20100101-20190101": (1262304000, 1546300800),
                  "20190101-20200906": (1546300800, 1599350400)}


def download_list_csv_to(symbol_list, destination, date_str):
    start, end = date_to_second[date_str]
    driver = webdriver.Chrome()
    if not os.path.exists(destination):
        os.makedirs(destination)
    for idx, sym in enumerate(symbol_list):
        print('\r', "Downloading " + str(idx + 1) + '/' + str(len(symbol_list)) + " files... ",
              end='')
        download_csv(driver, sym, str(start), str(end))
        timeout = time.time() + 2
        file_path = "C:/Users/Zhang Yuyao/Downloads/" + sym + ".csv"
        while True:
            if time.time() > timeout:
                break
            elif "400 Bad Request" in driver.page_source:
                break
            elif os.path.isfile(file_path):
                break
    print("Done.")
    time.sleep(1)
    for idx, sym in enumerate(symbol_list):
        print('\r', "Moving " + str(idx + 1) + '/' + str(len(symbol_list)) + " files... ",
              end='')
        file_path = "C:/Users/Zhang Yuyao/Downloads/" + sym + ".csv"
        if os.path.isfile(file_path):
            statinfo = os.stat(file_path)
            if statinfo.st_size < 300:
                os.remove(file_path)
            else:
                shutil.move(file_path, destination + sym + ".csv")
    print("Done.")


def download_csv(driver, sym, start, end):
    address = "https://query1.finance.yahoo.com/v7/finance/download/" + sym + "?period1=" + str(
        start) + "&period2=" + str(end) + "&interval=1d&events=history"
    driver.get(address);


def get_all_stock_symbols():
    f = open('data/stock_symbols_us.txt', 'r', encoding='UTF-8')
    arr = f.read().split("\n")
    for i in range(len(arr)):
        arr[i] = arr[i].strip(" \t").split('\t')[0]
    arr = list(set(arr))
    for i in range(len(arr))[::-1]:
        if not (arr[i].isupper() and arr[i].isalpha()):
            arr.pop(i)
    return sorted(arr)

