import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import xgboost as xgb
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.base import clone
import random
import matplotlib.pylab as plt

# TODO: Try a neural network on this problem
# TODO: Try logist regression on this problem

class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    def __len__(self):
        return self.len

class NeuralNet(nn.Module):
    def __init__(self, inputDim, hiddenDim, secondHiddenDim, outputDim)-> None:
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(inputDim, hiddenDim)
        self.fc2 = nn.Linear(hiddenDim, secondHiddenDim)
        self.fc3 = nn.Linear(secondHiddenDim, outputDim)
    
    def init_weights(self) -> None:

        for layer in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def forward(self, x) -> torch.Tensor:
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)

        return x

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


def transform_y_vals(y: npt.NDArray) -> npt.NDArray:
    return np.log1p(y)


def get_data(X: npt.NDArray, y: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    X = impute_missing_values(X)
    # X = normalize_feature_matrix(X)

    y = cleanup_weird_vals(y)

    return X, y

def get_splits_nn(X: npt.NDArray, y: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2
        )

    return X_train, X_test, y_train, y_test 


def print_score_info(scores, rmse_scores, mae_scores):
    print(f"Avg. R^2: {scores.mean()}")
    print(f"Standard deviation of scores: {scores.std()}")

    print(f"Avg. RMSE score: {rmse_scores.mean()}")
    print(f"Standard devation of RMSE scores: {rmse_scores.std()}")
    print(f"Avg. MAE score: {mae_scores.mean()}")

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

    # Chooses 20 random combination of the features to approximate optimal model while being more computationally efficient
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


    # Calculates every combination
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

def visualize_training(loss_vals):
    step = range(len(loss_vals))
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(step, np.array(loss_vals))
    plt.title("Loss at Each Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


def train_nn(model, train_dataLoader, learning_rate):
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=0.01)

    epochs = 200
    loss_vals = []

    for epoch in range(epochs):
        epoch_loss = 0
        for X, y in train_dataLoader:
            X = X.float()
            y = y.float()
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y.unsqueeze(-1))
            print(f"Epoch: {epoch}")
            print(f"Loss: {loss}")
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_dataLoader)
        loss_vals.append(avg_loss)
    visualize_training(loss_vals)

    return model

def evaluate_nn(model, test_dataLoader):
    # Write some code to test the predictions of model on unseen data
    y_actual = []
    y_pred = []
    n = 0

    with torch.no_grad():
        for X, y in test_dataLoader:
            X = X.float()
            y = y.float()

            pred = model(X)

            y_pred.append(pred)
            y_actual.append(y)
            n += 1

    y_actual = torch.cat(y_actual).detach().numpy()
    y_pred = torch.cat(y_pred).detach().numpy()

    y_pred = np.expm1(y_pred)
    y_actual = np.expm1(y_actual)

    mse = (1 / n) * np.sum((y_actual - y_pred)**2)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual, y_pred)

    print(f"RMSE: {rmse}")
    print(f"House predictions are typically about ${rmse} away from the actual house price.")
    print(f"R^2: {r2}")

    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], '--')
    plt.plot(y_actual, y_pred, 'o')
    plt.xlabel("Actual House Prices")
    plt.ylabel("Predicted House Prices")
    plt.title("Actual House Prices vs. Predict House Prices")
    plt.show()


# Predict on kth fold's test dataset and return r2 value for that fold
def eval_performance(model_trained, X_test, y_test):
    n = len(X_test)
    y_pred = model_trained.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = (1 / n) * np.sum((y_test - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = (1 / n) * np.sum(np.abs((y_test - y_pred)))

    return r2, rmse, mae


    

def kfolds(model, X, y, k):

    folds = RepeatedKFold(n_splits=k, n_repeats=10, random_state=42)

    r2Perf = []
    rmsePerf = []
    maePerf = []


    for i, (train_index, test_index) in enumerate(folds.split(X, y)):
        X_train = X[train_index]
        y_train = y[train_index]
        m = clone(model)
        m.fit(X_train, y_train)
        X_test = X[test_index]
        y_test = y[test_index]

        r2, rmse, mae = eval_performance(m, X_test, y_test)
        r2Perf.append(r2)
        rmsePerf.append(rmse)
        maePerf.append(mae)

    return np.array(r2Perf), np.array(rmsePerf), np.array(maePerf)




def main():


    housing_pd = pd.read_csv("./housing.csv")


    housing = housing_pd.to_numpy()
    print(housing)
    # graph_data(housing)
    # print(housing_pd)
    # housing = cleanup_weird_vals(housing)
    housing_pd = date_to_float(housing_pd)
    X = get_feature_vectors(housing_pd).to_numpy()
    y = housing_pd["sale_price"].to_numpy()

    X, y = get_data(X, y)
    # y = transform_y_vals(y)

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
    # mae_scores = -cross_val_score(mdl, X, y, cv=cv, scoring="neg_mean_absolute_error")

    # print_score_info(scores, rmse_scores, mae_scores)

    # depth, learning_rate, subsample, colsample = find_best_model(X, y)
    depth = 7
    learning_rate = 0.1
    subsample = 1
    colsample = 0.7

    best_model = xgb.XGBRegressor(n_estimators=100, max_depth=depth, eta=learning_rate, subsample=subsample, colsample_bytree=colsample)

    # cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

    # r2_scores = cross_val_score(best_model, X, y, cv=cv, scoring='r2')
    # rmse_scores = -cross_val_score(best_model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
    # mae_scores = -cross_val_score(best_model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    r2_scores, rmse_scores, mae_scores = kfolds(best_model, X, y, 5)
    print_score_info(r2_scores, rmse_scores, mae_scores)


    # X_train, X_test, y_train, y_test = get_splits_nn(X, y)

    # # Standardize X values 
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    # # Log transform y_values 
    # y_train = transform_y_vals(y_train)
    # y_test = transform_y_vals(y_test)

    # batch_size = 32

    # train_data = Data(X_train, y_train)
    # train_dataLoader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    # test_data = Data(X_test, y_test)
    # test_dataLoader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    # # Test to make sure splits are correct
    # # for batch, (X, y) in enumerate(train_dataLoader):
    # #     print(f"Batch: {batch+1}")
    # #     print(f"X shape: {X.shape}")
    # #     print(f"y shape: {y.shape}")
    # #     break




    # input_dim = 8
    # hidden_dim = 128
    # second_hidden_dim = 64
    # output_dim = 1
    # learning_rate = 0.001
    
    # neuralModel = NeuralNet(input_dim, hidden_dim, second_hidden_dim, output_dim)

    # model = train_nn(neuralModel, train_dataLoader, learning_rate)

    # evaluate_nn(model, test_dataLoader)



    


if __name__ == "__main__":
    main()
