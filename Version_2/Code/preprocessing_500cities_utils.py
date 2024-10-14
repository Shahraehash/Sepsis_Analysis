import pandas as pd
import copy
import os
import statsmodels.api as sm 
from itertools import combinations

NUMBER_OF_SEPSIS_PATIENTS_IN_2014 = 1700000

eICU_estimates_2014 = {
    "BPHIGH": 52.28,
    "CANCER": 18.66,
    "CASTHMA": 7.37,
    "CHD": 8.16,
    "COPD": 15.46,
    "DIABETES": 16.27,
}

RATE_OF_INCREASE_IN_SEPSIS = 2.47


def read_ids_into_list(path):
    # Get the Disease IDs we care about which were identified using clinical data
    with open(path, "r") as f:
        lines = f.readlines()
    ids = []
    for line in lines:
        ids += line.split()
    return ids


def clean_df(df, ids, pop_column, year):
    # Clean it so that we have consistent column names that we can work with
    df = df[df["MeasureId"].isin(ids)]
    df = df[df["StateAbbr"] != "US"]
    df = df[df["Year"] == int(year)]

    new_columns = ["Year", "StateAbbr", "CityName", "Data_Value", "MeasureId"]
    new_columns += [pop_column]

    # Also keep one column for location to make it unique
    state_tmp = df["StateAbbr"].copy()
    df["City,State"] = df["CityName"].str.cat(state_tmp, sep=", ")
    new_columns.remove("CityName")
    new_columns.remove("StateAbbr")
    new_columns += ["City,State"]

    # Calculate number of patients based on the metrics defined in the dataset
    df["Number_Of_Patients"] = df["Data_Value"] * df[pop_column] / 100
    df["Total_Patients"] = df.groupby(["City,State", "MeasureId"])["Number_Of_Patients"].transform("sum")
    new_columns.remove("Data_Value")
    new_columns.remove(pop_column)
    new_columns += ["Total_Patients"]
    df = df[new_columns]

    df = df.dropna()
    new_df = df.drop_duplicates(subset=["City,State", "MeasureId"])
    return new_df


def create_500_cities_df_per_year(df_path, ids_path, columns, year):
    # Function called for each year so this can be called for each year
    df = pd.read_csv(df_path)
    df = df[columns]
    ids = read_ids_into_list(ids_path)
    df = clean_df(df, ids, columns[-2], year)

    df_new = df.pivot(index="City,State", columns="MeasureId", values="Total_Patients")

    return df_new


def fix_dataframes_published(dict_of_dataframes):
    # Had to do this manually in order to have consistent data from year to year

    new_dict = {}
    df_2013 = dict_of_dataframes["2013"]
    df_2014 = dict_of_dataframes["2014"]
    df_2015 = dict_of_dataframes["2015"]
    df_2016 = dict_of_dataframes["2016"]
    df_2017 = dict_of_dataframes["2017"]
    df_2018 = dict_of_dataframes["2018"]
    df_2019 = dict_of_dataframes["2019"]

    # From testing found that the overlap present in all datasets is only in 2014-2019
    # and for only 366 cities (because of 2018) so will adjust for that below

    new_df_2014 = df_2014
    new_df_2014 = pd.merge(new_df_2014, df_2013, on="City,State")
    df_2016["BPHIGH"] = (df_2015["BPHIGH"] + df_2017["BPHIGH"]) / 2

    new_df_2018 = df_2018[df_2018.index.isin(df_2017.index)]

    new_df_2014 = new_df_2014[new_df_2014.index.isin(new_df_2018.index)]
    new_df_2015 = df_2015[df_2015.index.isin(new_df_2018.index)]
    new_df_2016 = df_2016[df_2016.index.isin(new_df_2018.index)]
    new_df_2017 = df_2017[df_2017.index.isin(new_df_2018.index)]
    new_df_2019 = df_2019[df_2019.index.isin(new_df_2018.index)]
    new_df_2018["BPHIGH"] = (new_df_2017["BPHIGH"] + new_df_2019["BPHIGH"]) / 2

    new_dict[2014] = new_df_2014
    new_dict[2015] = new_df_2015
    new_dict[2016] = new_df_2016
    new_dict[2017] = new_df_2017
    new_dict[2018] = new_df_2018
    new_dict[2019] = new_df_2019

    return new_dict


def estimate_sepsis_each_year(dict_of_dataframes):
    # Needed to estimate my y values since the data didnt provide tru y values. 

    # Scaled the number of estimated years with sepsis predicted in 2014 (using the markers denoted in literature)
    # to the true number of Sepsis patients in 2014. 
    df_2014 = dict_of_dataframes[2014]
    for column in dict_of_dataframes[2014].columns:
        if "SEPSIS" in dict_of_dataframes[2014].columns:
            df_2014["SEPSIS"] += (dict_of_dataframes[2014][column] * 1/(eICU_estimates_2014[column] / 100))
        else:
            df_2014["SEPSIS"] = (dict_of_dataframes[2014][column] * 1/ (eICU_estimates_2014[column] / 100))

    # Estimated the number of Sepsis patients in each consecutively year by using the rate of increase of SEPSIS 
    # patients published in literature and considering a linear trend. This is the biggest limitation and given 
    # a dataset with true SEPSIS patients and the data, this would eliminate error caused by this step
    new_dict = {}
    index = 0
    for key in sorted(dict_of_dataframes):
        dataframe = dict_of_dataframes[key]
        dataframe["SEPSIS"] = (df_2014["SEPSIS"] / df_2014["SEPSIS"].sum()) * (NUMBER_OF_SEPSIS_PATIENTS_IN_2014 * ((1 + RATE_OF_INCREASE_IN_SEPSIS / 100) ** index))
        new_dict[key] = dataframe
        index += 1

    return dict_of_dataframes


def create_500_cities_df(path_folder):
    # Main function performed for creating the dataframe
    dict_parameters = {
        "2016": [
            [
                "Year",
                "StateAbbr",
                "CityName",
                "Data_Value",
                "Population2010",
                "MeasureId",
            ],
            ["2013", "2014"],
        ],
        "2017": [
            [
                "Year",
                "StateAbbr",
                "CityName",
                "Data_Value",
                "Population2010",
                "MeasureId",
            ],
            ["2014", "2015"],
        ],
        "2018": [
            [
                "Year",
                "StateAbbr",
                "CityName",
                "Data_Value",
                "PopulationCount",
                "MeasureId",
            ],
            ["2015", "2016"],
        ],
        "2019": [
            [
                "Year",
                "StateAbbr",
                "CityName",
                "Data_Value",
                "PopulationCount",
                "MeasureId",
            ],
            ["2016", "2017"],
        ],
        "2020": [
            [
                "Year",
                "StateAbbr",
                "CityName",
                "Data_Value",
                "TotalPopulation",
                "MeasureId",
            ],
            ["2017", "2018"],
        ],
        "2021": [
            [
                "Year",
                "StateAbbr",
                "CityName",
                "Data_Value",
                "TotalPopulation",
                "MeasureId",
            ],
            ["2018", "2019"],
        ],
    }

    dict_of_dataframes = {}

    ids = "Data/500_Cities/Sepsis_published_ids.txt"

    for files in os.listdir(path_folder):
        if files.endswith(".csv"):
            for key in dict_parameters:
                if key in files:
                    for year in dict_parameters[key][1]:
                        if (year not in dict_of_dataframes or len(dict_of_dataframes[year].index) == 0):
                            dict_of_dataframes[year] = create_500_cities_df_per_year(path_folder + files, ids, dict_parameters[key][0], year)
                        else:
                            dataframe = create_500_cities_df_per_year(path_folder + files, ids, dict_parameters[key][0], year)
                            if len(dataframe.index) != 0:
                                for column in dataframe.columns:
                                    if column not in dict_of_dataframes[year]:
                                        dict_of_dataframes[year] = pd.merge(dict_of_dataframes[year],dataframe[column],on="City,State",)

    new_dict_of_dataframes = fix_dataframes_published(dict_of_dataframes)
    final_dict_dataframe = estimate_sepsis_each_year(new_dict_of_dataframes)

    return final_dict_dataframe


def concatenate_all_dataframes(dict_of_dataframes):
    # Concatenates all of the dataframes together
    return pd.concat(list(dict_of_dataframes.values()))


def AIC_significant(dataframe):
    # Only keeps the columns that have been shown to be significant through AIC testing done in R.
    
    sig_columns = [column for column in dataframe.columns if column != "SEPSIS"]
    all_combinations_of_columns = []
    for num_elems in range(1, len(sig_columns)+1):
        all_combinations_of_columns.extend(list(combinations(sig_columns, num_elems)))
    
    all_combinations = [[item for item in comb] for comb in all_combinations_of_columns]

    min_AIC = float("inf")
    best_columns = []
    for combination in all_combinations:
        X = dataframe[combination]
        y = dataframe["SEPSIS"]
        X = sm.add_constant(X)
        model = sm.OLS(y, X.astype(float)).fit()
        
        if model.aic < min_AIC:
            min_AIC = model.aic
            best_columns = combination
        
    best_columns += ["SEPSIS"]
    return dataframe[best_columns], best_columns


def SpearmanRank_significant(dataframe, n):
    # Runs a Spearman Rank test to identify the n most significant columns based on the dataset
    correlation = []
    for x in list(dataframe.corr()["SEPSIS"][:-1]):
        correlation += [abs(x)]

    best_feature = []
    for corr in correlation:
        if len(best_feature) < n:
            best_feature.append(max(correlation))
            correlation.remove(max(correlation))

    best = []
    for x in best_feature:
        if x in list(dataframe.corr()["SEPSIS"][:-1]):
            best.append(x)
        else:
            best.append(-x)

    best = [list(dataframe.corr()["SEPSIS"]).index(x) for x in best]
    columns = list(enumerate(dataframe.columns))
    significant_columns = [j[1] for x in best for j in columns if j[0] == x] + ["SEPSIS"]

    return dataframe[significant_columns], significant_columns
