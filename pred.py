import pandas as pd
import torch
import xgboost as xgb
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split



# Impute missing values with the average of that column
def impute_missing_values(X: npt.NDArray) -> npt.NDArray:
    means = np.nanmean(X[:, [0, 6, 7, 8, 9]], axis=0)

    cols = [0, 6, 7, 8, 9] # cols to impute 

    for idx in cols:
        means_idx = 0
        col = X[:, idx]
        col[pd.isnull(col)] = means[means_idx]
        means_idx += 1

    return X

def main():


    housing_pd = pd.read_csv("./housing.csv")


    housing = housing_pd.to_numpy()
    print(housing_pd)

    print(housing_pd.isna().sum())


    housing = impute_missing_values(housing)

    print(housing)


if __name__ == "__main__":
    main()
