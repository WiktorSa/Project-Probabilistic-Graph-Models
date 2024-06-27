import pandas as pd
from sklearn.model_selection import train_test_split


def split(train_size, val_size, test_size):
    data = pd.read_csv('data/input.csv')

    train, rest = train_test_split(data, test_size=val_size + test_size, stratify=data['y'], random_state=42, shuffle=True)
    val, test = train_test_split(rest, test_size=test_size / (val_size + test_size), stratify=rest['y'], random_state=42, shuffle=True)

    train.to_csv('data/split_train.csv', index=False)
    val.to_csv('data/split_val.csv', index=False)
    test.to_csv('data/split_test.csv', index=False)


if __name__ == '__main__':
    split(train_size=0.70, val_size=0.15, test_size=0.15)
