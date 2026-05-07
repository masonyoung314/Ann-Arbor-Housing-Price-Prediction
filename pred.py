import pandas as pd
import torch
import xgboost as xgb
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def date_to_float(df: pd.DataFrame) -> pd.DataFrame:
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["sale_date"] = df["sale_date"].astype("int64") / 1e9

    return df

# Impute missing values with the average of that column
def impute_missing_values(X: npt.NDArray) -> npt.NDArray:
    means = np.nanmean(X[:, [1, 2, 3, 4, 5, 6, 7]], axis=0)

    cols = [1, 2, 3, 4, 5, 6, 7] # cols to impute 

    for idx in cols:
        means_idx = 0
        col = X[:, idx]
        col[pd.isnull(col)] = means[means_idx]
        means_idx += 1

    return X


# Impute target value of rows with incorrect prices or no price
def cleanup_weird_vals(y: npt.NDArray) -> npt.NDArray:
    # If this doesn't work well, then I will consider dropping these rows and checking for model improvement
    mean = np.nanmean(y)

    y[pd.isnull(y)] = mean
    y[y < 50000] = mean

    return y

def get_feature_vectors(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["sale_date","beds","full_baths","half_baths","sqft","acres","lat","long"]]
    print(df)
    return df


def normalize_feature_matrix(X: npt.NDArray) -> npt.NDArray:
    # Normalize the feature matrix

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    return X



def get_data(X: npt.NDArray, y: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    X = impute_missing_values(X)
    X = normalize_feature_matrix(X)

    y = cleanup_weird_vals(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2
    )

    return X_train, X_test, y_train, y_test 

def main():


    housing_pd = pd.read_csv("./housing.csv")


    # housing = housing_pd.to_numpy()
    # print(housing_pd)
    # housing = cleanup_weird_vals(housing)
    housing_pd = date_to_float(housing_pd)
    X = get_feature_vectors(housing_pd).to_numpy()
    y = housing_pd["sale_price"].to_numpy()
    print(y)

    X_train, X_test, y_train, y_test = get_data(X, y)

    # print(f"# houses with missing price: {np.sum(pd.isnull(housing[:, 0]))}")
    # print(f"# houses with price < 5000: {np.size(np.where(housing[:, 0] < 5000))}")
    # print(f"# houses with price < 20000: {np.size(np.where(housing[:, 0] < 20000))}")
    # print(f"# houses with price < 50000: {np.size(np.where(housing[:, 0] < 50000))}")

    # with np.printoptions(threshold=np.inf):
    #     print(housing[housing[:, 0] < 50000])


    # print(housing_pd.isna().sum())



if __name__ == "__main__":
    main()
