import pandas as pd
import os

'''
GEn1E LifeSciences Fit n Grit Task:

We are interested in understanding the prevalence of Sepsis across 
the US. We would like you to use publicly available data to create 
a heat map across the US which showing the occurrence of Sepsis. 

'''

############################################# GLOBAL VARIABLES ###################################################

#Total Number of Sepsis patients diagnosed every year
TOTAL_SEPSIS_PATIENTS = 1700000

########### Bacteria Based Estimation Global Variables ############

#Which bacteria are gram positive and which are gram negative
BACTERIAL_TYPE = {"Gram Positive": ["Staphylococcus aureus", "Streptococcal"],
                  "Gram Negative": ["Escherichia coli"]}

#Which percentage of bacteria cause sepsis and which percentage of gram positive and gram negative
PERCENTAGE_BACTERIAL_SEPSIS = 0.241 + 0.285 
PERCENTAGE_SEPSIS_GRAM_POSITIVE = 0.622
PERCENTAGE_SEPSIS_GRAM_NEGATIVE = 0.468

#Bacteria Columns that are relevant
BACTERIA_COLUMNS = ["Reporting Area", "Vancomycin-resistant Staphylococcus aureus, Cum 2019†", 
                    "Streptococcal toxic shock syndrome, Cum 2019†", "Shiga toxin-producing Escherichia coli, Cum 2019†", 
                    "Sepsis Distribution by Bacteria"]

########### Disease Based Estimation Global Variables ##############

#Based on online report on the risk of sepsis given you have this disease
RISK_OF_SEPSIS = {"Chronic Lung Disease": 72.7/1000, "Peripheral Artery Disease": 68.5/1000, "Chronic Kidney Disease": 53.4/1000, 
                  "Myocardial Infarction": 60.2/1000, "Diabetes": 47.7/1000, "Stroke": 56.5/1000, 
                  "Deep Vein Thrombosis": 56.9/1000, "Coronary Artery Disease": 52.7/1000, "Hypertension": 37.6/1000, 
                  "Atrial Fibrillation": 49.4/1000, "Dyslipidemia": 38.2/1000}

#Disease
DISEASES_COLUMNS = ["StateDesc", "Chronic Kidney Disease", "Chronic Lung Disease", "Coronary Artery Disease", "Diabetes", "Hypertension", "Stroke", "Sepsis Distribution by Disease"]

################################################## FUNCTIONS ######################################################

def read_all_csvs_into_list(list_of_paths):
    '''
    Returns all pandas dataframes from paths provided.

    Args:
        list_of_paths: paths of each csv that need to be read -> [str, str, ...]
    
    Return:
        list_of_dataframes: all dataframes read by the paths provided -> [pd.DataFrame(), pd.DataFrame(), ...]
    '''
    list_of_dataframes = []
    for path in list_of_paths:
        list_of_dataframes += [pd.read_csv(path)]

    return list_of_dataframes

def get_list_of_states(path, upper):
    '''
    Returns list of all of the states based on provided .txt file of all U.S. states.

    Args: 
        path: path of the U.S. states file -> str
        upper: flag on whether to capitalize the U.S. states in list -> bool
    
    Return:
        us_states: list of all U.S. states in desired format -> [str, str, ...]
    '''
    with open(path) as f:
        us_states_from_file = f.readlines()
    us_states = []
    for state in us_states_from_file:
        if upper:
            us_states += [state[:-1].upper()]
        else:
            us_states += [state[:-1]]

    return us_states


def trim_dataframe(dataframe, columns, rows):
    '''
    Returns trimmed dataframe where rows provided are kept and duplicate rows are compressed and summed in column desired

    Args: 
        dataframe: dataframe that needs to be trimmed -> pd.DataFrame()
        columns: columns of the state and the value we care about -> [str, str]
        rows: the rows or states we want to keep -> [str, str, ...]
    
    Return:
       dataframe: trimmed dataframe after rows are compressed and kept -> pd.DataFrame()
    '''

    if rows != None:
        dataframe = dataframe.loc[dataframe[columns[0]].isin(rows)]
    if columns != None:
        dataframe = dataframe.groupby([columns[0]], as_index = False)[columns[1]].sum()

    return dataframe

def trim_all_dataframes(all_dfs, flag, rows):
    '''
    Returns all of the trimmed dataframes by calling trim_dataframe

    Args: 
        all_dfs: list of dataframes that need to be trimmed -> [pd.DataFrame(), pd.DataFrame(), ...]
        flag: flag for whether we are trimming bacteria dataframes or disease dataframes 
        since columns and rows are different then -> bool
        rows: the rows or states we want to keep -> [str, str, ...]
    
    Return:
       list_of_dataframes: list of trimmed dataframe after rows are compressed and kept -> [pd.DataFrame(), pd.DataFrame(), ...]
    '''

    list_of_dataframes = []
    for i, dataframe in enumerate(all_dfs):
        if flag:
            list_of_dataframes += [trim_dataframe(dataframe, [BACTERIA_COLUMNS[0], BACTERIA_COLUMNS[i+1]], rows)]
        else:
            list_of_dataframes += [trim_dataframe(dataframe, [DISEASES_COLUMNS[0], DISEASES_COLUMNS[i+1]], rows)]
    
    return list_of_dataframes

def merge_dataframes(list_of_dataframes, column):
    '''
    Returns the merged dataframes based on the states column so that we have a giant dataframe with statistics on each state

    Args: 
        list_of_dataframes: list of dataframes that need to be merged -> [pd.DataFrame(), pd.DataFrame(), ...]
        column: the column upon which we want to merge the dataframes -> str
    
    Return:
       final_df: list of merged dataframe -> [pd.DataFrame(), pd.DataFrame(), ...]
    '''

    final_df = list_of_dataframes[0]
    for i in range(1,len(list_of_dataframes)):
        final_df = pd.merge(final_df, list_of_dataframes[i], on = column, how = "inner")

    return final_df

def create_bacterial_df(final_merged_df):
    '''
    Returns final bacterial dataframe based on performing estimates with the merged dataframe and 
    using the percentage of gram positive and gram negative bacteria that lead to sepsis.

    Args: 
        final_merged_df: dataframe that has all of the data necessary for estimation -> pd.DataFrame()
    
    Return:
       final_merged_df: dataframe that only has the state and the estimation of number of patients in the state
                        that develop sepsis as a result of a bacterial infection -> pd.DataFrame()
    '''

    #Percentage of all Sepsis patients where the cause is a bacterial infection
    total_bacterial_patients = TOTAL_SEPSIS_PATIENTS * PERCENTAGE_BACTERIAL_SEPSIS

    #Percentage of each bateria if diagnosed with sepsis
    number_gram_positive = PERCENTAGE_SEPSIS_GRAM_POSITIVE * total_bacterial_patients
    number_gram_negative = PERCENTAGE_SEPSIS_GRAM_NEGATIVE * total_bacterial_patients

    #Create Gram Positive and Gram Negative Columns which can be weighted based on the chance they lead to Sepsis
    for bacteria_type in BACTERIAL_TYPE:
        final_merged_df[bacteria_type] = 0
        for bacteria in BACTERIAL_TYPE[bacteria_type]:
            for column in final_merged_df.columns:
                if bacteria in column:
                    if final_merged_df[column].sum() != 0:
                        if bacteria_type == "Gram Positive":
                            final_merged_df[bacteria_type] += final_merged_df[column]/final_merged_df[column].sum()*number_gram_positive
                        if bacteria_type == "Gram Negative":
                            final_merged_df[bacteria_type] += final_merged_df[column]/final_merged_df[column].sum()*number_gram_negative

    #Sum total number of patients that have sepsis to get one value per state
    final_merged_df[BACTERIA_COLUMNS[-1]] = final_merged_df["Gram Positive"] + final_merged_df["Gram Negative"]
    final_merged_df["state"] = final_merged_df[BACTERIA_COLUMNS[0]].str.lower()

    return final_merged_df[["state", BACTERIA_COLUMNS[-1]]]


def preprocessing_500_cities(list_of_dataframes):
    '''
    Returns a processed dataframe for 500 cities dataframes to anticipate errors in processing in the future. 

    Args: 
        list_of_dataframes: list of dataframes that need to be processed -> [pd.DataFrame(), pd.DataFrame(), ...]
    
    Return:
       final_list_of_dataframes: list of dataframes after processing is done -> [pd.DataFrame(), pd.DataFrame(), ...]
    '''

    final_list_of_dataframes = []
    for i, dataframe in enumerate(list_of_dataframes):
        #Change Data column so it's unique when we merge dataframes
        dataframe[DISEASES_COLUMNS[i+1]] = dataframe["Data_Value"]
        #For some reason 500 cities trimmed the North Carolina/South Carolina labels so fix them
        dataframe[DISEASES_COLUMNS[0]].replace('North Carolin', 'North Carolina', inplace=True)
        dataframe[DISEASES_COLUMNS[0]].replace('South Carolin', 'South Carolina', inplace=True)
        final_list_of_dataframes += [dataframe]

    return final_list_of_dataframes

def create_disease_df(final_merged_df):
    '''
    Returns final disease dataframe based on performing estimates with the merged dataframe and 
    using the risk of sepsis given a disease infection and normalizing it to equal the total number 
    of patients diagnosed with Sepsis annually

    Args: 
        final_merged_df: dataframe that has all of the data necessary for estimation -> pd.DataFrame()
    
    Return:
       final_merged_df: dataframe that only has the state and the estimation of number of patients in the state
                        that develop sepsis correlated with a specific disease -> pd.DataFrame()
    '''

    #Not all patients with disease cause Sepsis so get number of patients with Sepsis using indexed risk
    for disease in RISK_OF_SEPSIS:
        for column in final_merged_df.columns:
            if disease in column:
                final_merged_df[column] = final_merged_df[column] * RISK_OF_SEPSIS[disease]
    
    #Get a single value per state
    final_merged_df[DISEASES_COLUMNS[-1]] = 0
    for column in final_merged_df.columns:
        if column in DISEASES_COLUMNS[1:-1]:
            final_merged_df[DISEASES_COLUMNS[-1]] += final_merged_df[column]

    #Normalize so total patients equal the total annual number of Sepsis patients
    final_merged_df[DISEASES_COLUMNS[-1]] = final_merged_df[DISEASES_COLUMNS[-1]] * TOTAL_SEPSIS_PATIENTS/final_merged_df[DISEASES_COLUMNS[-1]].sum()

    final_merged_df["state"] = final_merged_df[DISEASES_COLUMNS[0]].str.lower()
    return final_merged_df[["state", DISEASES_COLUMNS[-1]]]


if __name__ == "__main__":
    current_path = os.getcwd()

    #Create list of states which can be used for estimation
    states_path = "/states.txt"
    list_of_states = get_list_of_states(current_path + states_path, False)
    list_of_states_upper = get_list_of_states(current_path + states_path, True)
    
    #Create bacterial dataframe
    staphylococcus_path = "/Bacterial_data/NNDSS_-_TABLE_1KK._Vancomycin-resistant_Staphylococcus_aureus_to_Varicella_morbidity.csv"
    streptococcus_path = "/Bacterial_data/NNDSS_-_TABLE_1HH._Streptococcal_toxic_shock_syndrome_to_Syphilis__Primary_and_Secondary.csv"
    ecoli_path = "/Bacterial_data/NNDSS_-_TABLE_1FF._Severe_acute_respiratory_syndrome-associated_coronavirus_disease_to_Shigellosis.csv"

    bacteria_dfs = read_all_csvs_into_list([current_path + staphylococcus_path, current_path + streptococcus_path, current_path + ecoli_path])
    trimmed_bacteria_dfs = trim_all_dataframes(bacteria_dfs, True, list_of_states_upper)
    final_merged_bacteria_df = merge_dataframes(trimmed_bacteria_dfs, BACTERIA_COLUMNS[0])
    bacterial_df = create_bacterial_df(final_merged_bacteria_df)
    bacterial_df.to_csv(current_path + "/bacterial_distribution_df.csv")
    

    #Create disease dataframe
    kidney_disease_path = "/Correlated_diseases/500_Cities__Chronic_kidney_disease_among_adults_aged___18_years.csv"
    lung_disease_path = "/Correlated_diseases/500_Cities__Chronic_obstructive_pulmonary_disease_among_adults_aged___18_years.csv"
    heart_disease_path = "/Correlated_diseases/500_Cities__Coronary_heart_disease_among_adults_aged___18_years.csv"
    diabetes_path = "/Correlated_diseases/500_Cities__Diagnosed_diabetes_among_adults_aged___18_years.csv"
    hypertension_path = "/Correlated_diseases/500_Cities__High_blood_pressure_among_adults_aged___18_years.csv"
    stroke_path  = "/Correlated_diseases/500_Cities__Stroke_among_adults_aged___18_years.csv"

    disease_dfs = read_all_csvs_into_list([current_path + kidney_disease_path, current_path + lung_disease_path, current_path + heart_disease_path, current_path + diabetes_path, current_path + hypertension_path, current_path + stroke_path])
    post_processed_disease_dfs = preprocessing_500_cities(disease_dfs)        
    trimmed_disease_dfs = trim_all_dataframes(post_processed_disease_dfs, False, list_of_states)
    final_merged_disease_df = merge_dataframes(trimmed_disease_dfs, DISEASES_COLUMNS[0])
    disease_df = create_disease_df(final_merged_disease_df)
    disease_df.to_csv(current_path + "/disease_distribution_df.csv")
    

