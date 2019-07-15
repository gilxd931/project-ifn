import pandas as pd


class CsvConverter:
    @staticmethod
    def convert(csv_file_path):
        df = pd.read_csv(csv_file_path)
        y = df['Class'].values
        x = []
        df = df.drop("Class", axis=1)
        cols = list(df.columns.values)
        for index, row in df.iterrows():
            record = [elem for elem in row]
            x.append(record)

        return x, y, cols
