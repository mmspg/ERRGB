import os
import pandas as pd


def files_2_csv(datapath):
    df = pd.DataFrame([])
    for file in os.listdir(datapath):
        if file == "Thumbs.db" or file == ".DS_Store":
            continue
        id = file.split("_")[0]
        id = id.lstrip("0")
        df = df.append(pd.DataFrame({'filename': file, 'id': id}, index=[0]), ignore_index=True)

    df.to_csv(os.path.join(os.getcwd(), "datasets", "query.csv"))


if __name__ == "__main__":
    datapath = os.path.join(os.getcwd(), "datasets", "query")
    files_2_csv(datapath)
