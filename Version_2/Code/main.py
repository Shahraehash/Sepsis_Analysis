from preprocessing_500cities_utils import *
from model_utils import *
from graph_utils import *
from estimating_sepsis_regression import *



if __name__ == "__main__":
    print("Creating all Dataframes")
    dict_of_dataframes = create_500_cities_df("Data/500_Cities/")
    concat_dataframe = concatenate_all_dataframes(dict_of_dataframes)

    # Testing different parameter orientation to reach the best model for the data
    tt_random_models = train_test_random(concat_dataframe)
    sig_models = significant_models(concat_dataframe)
    
    all_models = tt_random_models
    all_models.update(sig_models)
    
    # This was the hope to see if the model could perform well by estimating a year's 
    # datapoints without any knowledge of it, but it was a side experiment, not something 
    # that will be used in the final model.
    leave_one_year(dict_of_dataframes)
 
    # Estimate each independent variable for the next 10 years
    print("\nEstimating variables for the next " + str(NUM_ESTIMATED_YEARS) + " years.")
    dict_estimations, year_list = estimate_variables(dict_of_dataframes, NUM_ESTIMATED_YEARS)
    all_regression_estimated_dict = regression_estimate_sepsis(dict_estimations, year_list, all_models)

    print("\nGraphing Results")
    # Graph results
    actual_years = list(dict_of_dataframes.keys())
    predicted_years = [years for years in year_list if years not in actual_years]

    Sepsis_columns = []
    for key in all_regression_estimated_dict:
        Sepsis_columns = [column for column in all_regression_estimated_dict[key].columns if "SEPSIS" in column]
        break

    graph_sepsis_patients_over_years(all_regression_estimated_dict, Sepsis_columns, actual_years, predicted_years)
    graph_sepsis_patients_over_years(all_regression_estimated_dict, ["SEPSIS", "SEPSIS_pred_all_random_test_RF", "SEPSIS_pred_AIC_RF", "SEPSIS_pred_SR_RF"], actual_years, predicted_years)
    graph_most_regions_over_years(all_regression_estimated_dict, Sepsis_columns, actual_years, predicted_years, 5, YEAR_OF_INTEREST)  
    
