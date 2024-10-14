from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

NUMBER_OF_FOLDS = 5

ORDER_OF_MODELS = ["Linear", "Lasso", "Ridge", "RF", "XGB"]


def simple_linear_regression(X_train, X_test, y_train, y_test):
    # Linear Regression Model
    linear_regression = LinearRegression()
    cv_score = np.mean(
        cross_val_score(linear_regression, X_train, y_train, cv=NUMBER_OF_FOLDS)
    )
    linear_regression.fit(X_train, y_train)
    y_pred = linear_regression.predict(X_test)
    r_squared = r2_score(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred, squared=False) / (
        np.max(y_test) - np.min(y_test)
    )
   
    print("Linear Regression R^2: %.3f MSE: %.3f CV_Score: %.3f" % (r_squared, MSE, cv_score))
    return linear_regression


def lasso_regression(X_train, X_test, y_train, y_test):
    # Lasso Regression Model
    lasso_regression = Lasso(normalize=True)
    cv_score = np.mean(
        cross_val_score(lasso_regression, X_train, y_train, cv=NUMBER_OF_FOLDS)
    )
    lasso_regression.fit(X_train, y_train)
    y_pred = lasso_regression.predict(X_test)
    r_squared = r2_score(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred, squared=False) / (
        np.max(y_test) - np.min(y_test)
    )

    print("Lasso Regression R^2: %.3f MSE: %.3f CV_Score: %.3f" % (r_squared, MSE, cv_score))
    return lasso_regression


def ridge_regression(X_train, X_test, y_train, y_test):
    # Ridge Regression Model
    ridge_regression = Ridge()
    cv_score = np.mean(
        cross_val_score(ridge_regression, X_train, y_train, cv=NUMBER_OF_FOLDS)
    )
    ridge_regression.fit(X_train, y_train)
    y_pred = ridge_regression.predict(X_test)
    r_squared = r2_score(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred, squared=False) / (
        np.max(y_test) - np.min(y_test)
    )

    print("Ridge Regression R^2: %.3f MSE: %.3f CV_Score: %.3f" % (r_squared, MSE, cv_score))
    return ridge_regression


def random_forest_regression(X_train, X_test, y_train, y_test):
    # Random Forest Regression
    random_forest = RandomForestRegressor()
    cv_score = np.mean(
        cross_val_score(random_forest, X_train, y_train, cv=NUMBER_OF_FOLDS)
    )
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    r_squared = r2_score(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred, squared=False) / (
        np.max(y_test) - np.min(y_test)
    )

    print("Random Forest Regression R^2: %.3f MSE: %.3f CV_Score: %.3f" % (r_squared, MSE, cv_score))
    return random_forest


def XGB_regression(X_train, X_test, y_train, y_test):
    # XGB Regression
    xgb = XGBRegressor()
    cv_score = np.mean(cross_val_score(xgb, X_train, y_train, cv=NUMBER_OF_FOLDS))
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    r_squared = r2_score(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred, squared=False) / (
        np.max(y_test) - np.min(y_test)
    )

    print("XGB Regression R^2: %.3f MSE: %.3f CV_Score: %.3f" % (r_squared, MSE, cv_score))
    return xgb


def arrange_dataframes_by_variable(dict_of_dataframe):
    # Rearrangement of the dataframes so that the keys are the variables
    rownames = []
    new_dataframe_dict = {}
    for key in dict_of_dataframe:
        if rownames == []:
            rownames = list(dict_of_dataframe[key].index)
        for column_name, column in dict_of_dataframe[key].items():
            new_dataframe_dict[str(column_name) + "_" + str(key)] = column.tolist()

    dict_of_dictionaries = {}
    for key in new_dataframe_dict:
        keys = key.split("_")
        new_key = keys[1]
        if keys[0] not in dict_of_dictionaries:
            dict_of_dictionaries[keys[0]] = {new_key: new_dataframe_dict[key]}
        else:
            new_dict = {new_key: new_dataframe_dict[key]}
            dict_of_dictionaries[keys[0]].update(new_dict)

    new_dict_of_dataframes = {}
    for key in dict_of_dictionaries:
        dataframe = pd.DataFrame.from_dict(dict_of_dictionaries[key])
        dataframe.index = rownames
        new_dict_of_dataframes[key] = dataframe

    return new_dict_of_dataframes


def estimate_dataframe(dataframe, num_years):
    # Estimate the dataframe for the number of years provided
    num_years_for_estimation = len(dataframe.columns)
    min_year = min(dataframe.columns)
    diff_years = num_years - num_years_for_estimation
    estimate_dictionary = {}

    # Performs a linear regression for each row so it treats each region 
    # independent of the other and can estimate what the value for the region 
    # will be next x years.
    model = LinearRegression()
    for index, row in dataframe.iterrows():
        y = row.to_numpy().reshape(-1, 1)
        X = np.arange(y.size).reshape(-1, 1)
        X_pred = np.arange(diff_years).reshape(-1, 1)
        X_pred += num_years_for_estimation
        model.fit(X, y)
        y_pred = model.predict(X_pred)
        combined_y = np.concatenate((y, y_pred), axis=0).flatten()
        estimate_dictionary[index] = combined_y.tolist()

    dataframe = pd.DataFrame.from_dict(estimate_dictionary)

    years = [int(min_year) + x for x in range(num_years)]
    dataframe.index = years
    t_dataframe = dataframe.T

    return t_dataframe, years


def estimate_variables(dict_of_dataframe, years):
    # Master function that calls estimate_dataframe so that it can call everyvariable individually for estimation
    year_arranged_dict = arrange_dataframes_by_variable(dict_of_dataframe)
    list_year = []
    estimated_dict_dataframe = {}
    for key in year_arranged_dict:
        dataframe, years_list = estimate_dataframe(year_arranged_dict[key], years)
        estimated_dict_dataframe[key] = dataframe
        if list_year == []:
            list_year = years_list
    return estimated_dict_dataframe, list_year


def arrange_dataframes_by_year(dict_of_dataframe, list_of_years):
    # Rearrange dictionary of dataframes done by year into one where the keys are the years
    new_dict_of_dataframes = {}
    for year in list_of_years:
        dictionary_of_elems = {}
        diseases = []
        for key in dict_of_dataframe:
            dictionary_of_elems[key] = dict_of_dataframe[key][year].tolist()
            diseases += [key]
        new_dataframe = pd.DataFrame.from_dict(dictionary_of_elems)

        new_dataframe.index = dict_of_dataframe[key].index
        new_dict_of_dataframes[year] = new_dataframe

    return new_dict_of_dataframes


def regression_estimate_sepsis(dict_of_dataframe, list_of_years, dict_of_models):
    # Use the models to estimate the number of Sepsis patients using the data for that 
    # given year based on data for that year. 

    new_dict_of_dataframe = arrange_dataframes_by_year(dict_of_dataframe, list_of_years)
    updated_dict_of_dataframes = {}
    for key_year in new_dict_of_dataframe:
        dataframe = new_dict_of_dataframe[key_year].copy()
        for key_model in dict_of_models:
            mod_dataframe = dataframe.copy()
            if len(dict_of_models[key_model]) > 1:
                mod_dataframe = mod_dataframe[dict_of_models[key_model][0]]
            mod_dataframe = mod_dataframe.loc[:, mod_dataframe.columns != "SEPSIS"]
            for model in dict_of_models[key_model][-1]:
                new_column = ("SEPSIS_pred_" + str(key_model) + "_" + ORDER_OF_MODELS[(dict_of_models[key_model][-1]).index(model)])
                dataframe[new_column] = model.predict(mod_dataframe)

        updated_dict_of_dataframes[key_year] = dataframe

    return updated_dict_of_dataframes
