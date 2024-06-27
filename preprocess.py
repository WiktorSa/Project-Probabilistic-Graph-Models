import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, KBinsDiscretizer


Y_MAPPING = {
    'Underweight': 0,
    'Normal': 1,
    'Overweight I': 2,
    'Overweight II': 3,
    'Obesity I': 4,
    'Obesity II': 5,
    'Obesity III': 6,
}

def _apply(transformer, train: pd.Series, val: pd.Series, test: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    train = transformer.fit_transform(pd.DataFrame(train))
    val = transformer.transform(pd.DataFrame(val))
    test = transformer.transform(pd.DataFrame(test))
    
    # Print so you can later copy all values into Bayesian Neural Network
    if isinstance(transformer, OrdinalEncoder):
        print(transformer.categories_)
    
    return train, val, test


def preprocess(categorical_to_numerical_scale: bool = True,
               categorical_to_one_hot: bool = False,
               continous_to_discrete: bool = True):
    train = pd.read_csv('data/split_train.csv')
    val = pd.read_csv('data/split_val.csv')
    test = pd.read_csv('data/split_test.csv')

    cols_categorical = list(train.select_dtypes(include=['object', 'category']).columns)
    cols_numerical = list(train.select_dtypes(include=['number']).columns)

    assert not all([categorical_to_numerical_scale, categorical_to_one_hot]), \
        'Only one version of categorical convertion could be selected'

    train['y'] = train['y'].map(Y_MAPPING).astype(int)
    val['y'] = val['y'].map(Y_MAPPING).astype(int)
    test['y'] = test['y'].map(Y_MAPPING).astype(int)

    if categorical_to_numerical_scale:
        cols_numerical += cols_categorical
        for c in cols_categorical:
            if c == 'y': continue
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
            train[c], val[c], test[c] = _apply(encoder, train[c], val[c], test[c])
            train[c], val[c], test[c] = train[c].fillna(0), val[c].fillna(0), test[c].fillna(0)
            train[c], val[c], test[c] = train[c].astype(int), val[c].astype(int), test[c].astype(int)

    if categorical_to_one_hot:
        cols_numerical += cols_categorical
        for c in cols_categorical:
            if c == 'y': continue
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            train_c, val_c, test_c = _apply(encoder, train[c], val[c], test[c])
            for stage, stage_c in zip([train, val, test], [train_c, val_c, test_c]):
                for i in range(stage_c.shape[1]):
                    stage[f'{c}_{i}'] = stage_c[:,i].astype(int)
                stage.drop(c, axis=1, inplace=True)

    if continous_to_discrete:
        for c in cols_numerical:
            if c == 'y': continue
            discretizer = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
            train[c], val[c], test[c] = _apply(discretizer, train[c], val[c], test[c])
            train[c], val[c], test[c] = train[c].astype('category'), val[c].astype('category'), test[c].astype('category')
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
            train[c], val[c], test[c] = _apply(encoder, train[c], val[c], test[c])
            train[c], val[c], test[c] = train[c].fillna(0), val[c].fillna(0), test[c].fillna(0)
            train[c], val[c], test[c] = train[c].astype(int), val[c].astype(int), test[c].astype(int)

    train.to_csv('data/preprocessed_train.csv', index=False)
    val.to_csv('data/preprocessed_val.csv', index=False)
    test.to_csv('data/preprocessed_test.csv', index=False)


if __name__ == '__main__':
    preprocess()
