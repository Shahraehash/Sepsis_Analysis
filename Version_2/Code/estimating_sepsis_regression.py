from preprocessing_500cities_utils import *
from model_utils import *
from sklearn.model_selection import train_test_split

SEED = 28

NUM_ESTIMATED_YEARS = 15

YEAR_OF_INTEREST = 2023

def run_models(X_train, X_test, y_train, y_test):
    
    # Linear Models
    linear_model = simple_linear_regression(X_train, X_test, y_train, y_test)
    lasso_model = lasso_regression(X_train, X_test, y_train, y_test)
    ridge_model = ridge_regression(X_train, X_test, y_train, y_test)
    
    # Non-Linear Models
    random_forest_model = random_forest_regression(X_train, X_test, y_train, y_test)
    XGB_model = XGB_regression(X_train, X_test, y_train, y_test)
    return [linear_model, lasso_model, ridge_model, random_forest_model, XGB_model]


def train_test_random(concat_dataframe):
    # Randomly select a train and a test set across all of the years and use this to train the models. 
    print("\nPerforming Random Train/Test Experiment...")
    X = concat_dataframe.iloc[:,:-1].to_numpy()
    y = concat_dataframe.iloc[:,-1].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = SEED)
    trained_models = run_models(X_train, X_test, y_train, y_test)
    return {"all_random_test": [trained_models]}


def leave_one_year(dict_of_dataframes):
    # My curiosity considered the idea that what if we tested the model's ability to predict values for 
    # one year if it was trained on all of the other years. This might give some idea of the model's ability 
    # to predict years that it doesnt have any data about. 
    print("\nPerforming Test On One Year and Train on Remaining Experiment...")
    year_dict = {}
    for key in dict_of_dataframes:
        print("Year: ", key)
        dict_copy = dict_of_dataframes.copy()
        test = dict_copy.pop(key)
        train = concatenate_all_dataframes(dict_copy)

        X_test = test.iloc[:,:-1].to_numpy()
        y_test = test.iloc[:,-1].to_numpy()
        X_train = train.iloc[:,:-1].to_numpy()
        y_train = train.iloc[:,-1].to_numpy()

        trained_models = run_models(X_train, X_test, y_train, y_test)
        year_dict["leave_" + str(key) + "_out"] = [trained_models]
    return year_dict

def significant_models(concat_dataframe):
    # Use different metrics to remove columns that are insignificant in estimating number of Sepsis patients. 
    print("\nAssessing Model Only Looking at Significant Variables...")
    print("Looking at AIC Significant Columns")
    AIC_dataframe, AIC_columns = AIC_significant(concat_dataframe)
    X_AIC = AIC_dataframe.iloc[:,:-1].to_numpy()
    y_AIC = AIC_dataframe.iloc[:,-1].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X_AIC, y_AIC, test_size = 0.2, random_state = SEED)
    AIC_models = run_models(X_train, X_test, y_train, y_test)
    
    print("\n")
    
    print("Looking at Spearman Index Significant Columns")
    SR_dataframe, SR_columns = SpearmanRank_significant(concat_dataframe, 4)
    X_SR = SR_dataframe.iloc[:,:-1].to_numpy()
    y_SR = SR_dataframe.iloc[:,-1].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X_SR, y_SR, test_size = 0.2, random_state = SEED)
    SR_models = run_models(X_train, X_test, y_train, y_test)
    return {"AIC": [AIC_columns, AIC_models], "SR": [SR_columns, SR_models]}


