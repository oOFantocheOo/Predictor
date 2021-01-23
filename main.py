from train import run


def main():
    run(bar=0.0, folder_path="data/20100101-20190101/", save_model=True)
    run(bar=0.02, folder_path="data/20100101-20190101/", save_model=True)
    run(bar=0.03, folder_path="data/20100101-20190101/", save_model=True)


main()
