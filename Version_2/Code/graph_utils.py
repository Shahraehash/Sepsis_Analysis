import matplotlib.pyplot as plt
import numpy as np

def graph_scatter_plot(x_values, y_values, top_title, y_label, x_label, filepath):
    # Performs a scatter plot given values and does it with the labels/titles/filename as provided
    for key in y_values:
        plt.plot(x_values, y_values[key], label = key)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(top_title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(filepath, bbox_inches='tight')
    plt.clf()
    return 


def graph_sepsis_patients_over_years(all_dict_of_dataframes, Sepsis_columns, act_years, pred_years):
    # Gets the total number of sepsis patients using all of the regression techniques for each year to create X values
    dict_datapoints = {}
    for year in act_years:
        for column in all_dict_of_dataframes[year].columns:
            if column in Sepsis_columns:
                if column not in dict_datapoints:
                    dict_datapoints[column] = [(all_dict_of_dataframes[year])["SEPSIS"].sum()]
                else:
                    dict_datapoints[column] += [(all_dict_of_dataframes[year])["SEPSIS"].sum()]

    for year in pred_years:
        for column in all_dict_of_dataframes[year].columns:
            if column in Sepsis_columns:
                total_patients = (all_dict_of_dataframes[year])[column].sum()
                if int(year) == 2023:
                    print("The total number of Sepsis patients in 2023 for model ", column, " is estimated to ", total_patients)
                dict_datapoints[column] += [total_patients]

    graph_scatter_plot(act_years + pred_years, dict_datapoints, "Total Number of Patients with Sepsis Estimated", "Number of Patients", "Year", "Images/Total_Sepsis_patients_estimated_" + str(len(Sepsis_columns))+ "_columns.jpg")

    new_dict_datapoints = {}
    for key in dict_datapoints:
        new_dict_datapoints[key] = np.log10(dict_datapoints[key])
    graph_scatter_plot(act_years + pred_years, new_dict_datapoints, "Total Number of Patients with Sepsis Estimated", "Log Number of Patients", "Year", "Images/Log_Total_Sepsis_patients_estimated_" + str(len(Sepsis_columns))+ "_columns.jpg")
    return

def identify_top_regions(dataframe, column, number):
    # Identifies the top x regions in the SEPSIS column estimated 
    dataframe = dataframe.nlargest(number, column)
    return dataframe.index.values.tolist()

def graph_most_regions_over_years(all_dict_of_dataframes, Sepsis_columns, act_years, pred_years, num_locations, year_of_interest):
    # Graphs the top regions number of sepsis patients across the years 
    num_Sepsis = ['SEPSIS', 'SEPSIS_pred_all_random_test_RF', 'SEPSIS_pred_AIC_RF', 'SEPSIS_pred_SR_RF']

    for sepsis_column in num_Sepsis:
        regions = identify_top_regions(all_dict_of_dataframes[year_of_interest], sepsis_column, num_locations)
        print("The top ", num_locations, " regions in ", year_of_interest, " are as follows: ", regions, " with the number of patients as follows: \n", (all_dict_of_dataframes[year_of_interest]).loc[regions, sepsis_column])
        
        for region in regions:
            dict_datapoints = {}
            for year in act_years:
                for column in all_dict_of_dataframes[year].columns:
                    if column in Sepsis_columns:
                        if column not in dict_datapoints:
                            dict_datapoints[column] = [(all_dict_of_dataframes[year])["SEPSIS"].loc[region]]
                        else:
                            dict_datapoints[column] += [(all_dict_of_dataframes[year])["SEPSIS"].loc[region]]

            for year in pred_years:
                for column in all_dict_of_dataframes[year].columns:
                    if column in Sepsis_columns:
                        dict_datapoints[column] += [(all_dict_of_dataframes[year])[column].loc[region]]

            graph_scatter_plot(act_years + pred_years, dict_datapoints, "Number of Patients with Sepsis Estimated in " + str(region), "Number of Patients", "Year", "Images/" + str(sepsis_column) + "_patients_estimated_in_" + str(region)+".jpg")

    return 


