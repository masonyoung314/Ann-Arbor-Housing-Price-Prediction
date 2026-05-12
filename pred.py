import pandas as pd
import torch
import xgboost as xgb
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
import random
import matplotlib.pylab as plt

# TODO: Try a neural network on this problem
# TODO: Try logist regression on this problem

def graph_data(data: npt.NDArray):
    plt.plot(data[:, 3], data[:, 0], 'o')
    plt.show()

# Convert year-month-day to float of seconds since 1970
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
    # print(df)
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

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X,
    #     y,
    #     test_size=0.2
    # )

    # return X_train, X_test, y_train, y_test 

    return X, y

def print_score_info(scores, rmse_scores):
    print(f"Scores raw values: {scores}")
    print(f"Mean of scores: {scores.mean()}")
    print(f"Standard deviation of scores: {scores.std()}")

    print(f"Mean of RMSE scores: {rmse_scores.mean()}")
    print(f"Standard devation of RMSE scores: {rmse_scores.std()}")

def find_best_model(X, y):
    # Randomized grid search to find best XGBRegressor hyperparameters
    # Test 20 random combinations, take the best one
    max_depths = [3, 5, 7]
    n_estimators = 200
    learning_rates = [0.03, 0.05, 0.1]
    subsamples = [0.7, 0.8, 1]
    colsample_bytrees = [0.7, 0.8, 1]

    best_combination = None
    best_mean = None
    
    for i in range(0, 21):
        depth = random.choice(max_depths)
        learning_rate = random.choice(learning_rates)
        subsample = random.choice(subsamples)
        colsample = random.choice(colsample_bytrees)

        model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=depth, eta=learning_rate, subsample=subsample, colsample_bytree=colsample)
        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

        if i == 0 or (scores.mean() > best_mean):
            best_combination = depth, learning_rate, subsample, colsample
            best_mean = scores.mean()

    # for max_depth in max_depths:
    #     for learning_rate in learning_rates:
    #         for subsample in subsamples:
    #             for colsample in colsample_bytrees:
    #                 model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, eta=learning_rate, subsample=subsample, colsample_bytree=colsample)
    #                 cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    #                 scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

    #                 if best_combination == None or (scores.mean() < best_mean):
    #                     best_combination = max_depth, learning_rate, subsample, colsample
    #                     best_mean = scores.mean()

    print("Best combination of hyperparameters: ")
    print(f"Max_Depth: {best_combination[0]}")
    print(f"Eta: {best_combination[1]}")
    print(f"Subsample: {best_combination[2]}")
    print(f"Colsample_bytree: {best_combination[3]}")
    print(f"Best mean R^2: {best_mean}")

    return best_combination[0], best_combination[1], best_combination[2], best_combination[3]




def main():


    housing_pd = pd.read_csv("./housing.csv")


    housing = housing_pd.to_numpy()
    print(housing)
    graph_data(housing)
    # print(housing_pd)
    # housing = cleanup_weird_vals(housing)
    housing_pd = date_to_float(housing_pd)
    X = get_feature_vectors(housing_pd).to_numpy()
    y = housing_pd["sale_price"].to_numpy()

    # X_train, X_test, y_train, y_test = get_data(X, y)
    X, y = get_data(X, y)

    # mdl = xgb.XGBRegressor(n_estimators=200, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)

    # cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)

    # scores = cross_val_score(mdl, X, y, cv=cv, scoring='r2')
    # rmse_scores = -cross_val_score(
    #     mdl,
    #     X,
    #     y,
    #     scoring='neg_root_mean_squared_error',
    #     cv=cv
    # )

    # print_score_info(scores, rmse_scores)

    # depth, learning_rate, subsample, colsample = find_best_model(X, y)

    # best_model = xgb.XGBRegressor(n_estimators=1000, max_depth=depth, eta=learning_rate, subsample=subsample, colsample_bytree=colsample)

    # cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

    # scores = cross_val_score(best_model, X, y, cv=cv, scoring='r2')
    # rmse_scores = -cross_val_score(best_model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
    # print_score_info(scores, rmse_scores)

    


if __name__ == "__main__":
    main()
