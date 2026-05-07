import pandas as pd
import torch
import xgboost as xgb
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



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


# Impute target value of rows with incorrect prices or no price
def cleanup_weird_vals(arr: npt.NDArray) -> npt.NDArray:
    # If this doesn't work well, then I will consider dropping these rows and checking for model improvement
    mean = np.nanmean(arr[:, 0])

    arr[pd.isnull(arr[:, 0])] = mean
    arr[arr[:, 0] < 50000] = mean

    return arr

def get_feature_vectors(df: pd.DataFrame) -> pd.DataFrame:

    return df


def normalize_feature_matrix(X: npt.NDArray) -> npt.NDArray:
    # Normalize the feature matrix
    
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    return X



def get_data(X: npt.NDArray, y: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    housing = impute_missing_values(housing)

    X 
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2
    )

    return X_train, X_test, y_train, y_test 

def main():


    housing_pd = pd.read_csv("./housing.csv")


    housing = housing_pd.to_numpy()
    # print(housing_pd)
    housing = cleanup_weird_vals(housing)

    X_train, X_test, y_train, y_test = get_data(housing)

    # print(f"# houses with missing price: {np.sum(pd.isnull(housing[:, 0]))}")
    # print(f"# houses with price < 5000: {np.size(np.where(housing[:, 0] < 5000))}")
    # print(f"# houses with price < 20000: {np.size(np.where(housing[:, 0] < 20000))}")
    # print(f"# houses with price < 50000: {np.size(np.where(housing[:, 0] < 50000))}")

    # with np.printoptions(threshold=np.inf):
    #     print(housing[housing[:, 0] < 50000])


    # print(housing_pd.isna().sum())



if __name__ == "__main__":
    main()
