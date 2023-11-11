#find important features to not generate too much features

from autogluon.tabular import TabularDataset, TabularPredictor

import pandas as pd
from itertools import combinations
import math
import numpy as np

def main():



    pd.options.mode.chained_assignment = None

    train_data = TabularDataset('A/preproc_train_observed_A.csv')
    train_data.head()
    label = 'pv_measurement'
    test_data = TabularDataset(f'A/preproc_train_estimated_A.csv')

    filter1 = [col for col in train_data.columns if "snow" not in col]
    filter2 = [col for col in test_data.columns if "snow" not in col]

    

    train_data = train_data[filter1]
    test_data = test_data[filter2]

    #train_data = train_data.drop("wind_speed_w_1000hPa:ms",axis=1)
    #test_data = test_data.drop("wind_speed_w_1000hPa:ms",axis=1)
    #train_data = train_data.drop("elevation:m",axis=1)
    #test_data = test_data.drop("elevation:m",axis=1)

    

    percentage = 0.50
    num_rows = int(len(test_data) * percentage)
    sample = test_data.sample(n=num_rows, random_state=42)
    test_data = test_data.drop(sample.index)
    #percentage2 = 0.8
    #num_rows2 = int(len(train_data) * percentage2)
    #train_data = train_data.sample(n=num_rows2, random_state=42)
    train_data = pd.concat([train_data,sample]).reset_index(drop=True)

    train_data = create_wind(train_data)
    test_data = create_wind(test_data)


    amount_features_1 = 15
    amount_features_final = 37

    predictor = TabularPredictor(label=label, eval_metric='mean_absolute_error').fit(train_data, tuning_data=test_data)

    x = predictor.feature_importance(test_data)

    print(x)


    submission_data = TabularDataset('A/preproc_test_estimated_A.csv')
    submission_data = create_wind(submission_data)
    #submission_data = submission_data.drop("wind_speed_w_1000hPa:ms",axis=1)
    #submission_data = submission_data.drop("elevation:m",axis=1)


    predictions_A = predictor.predict(submission_data)

    ########

    predictions_A.to_csv('temp_a.csv')

    ##########B

    train_data = TabularDataset('B/preproc_train_observed_B.csv')
    label = 'pv_measurement'
    test_data = TabularDataset(f'B/preproc_train_estimated_B.csv')

    percentage = 0.50
    num_rows = int(len(test_data) * percentage)
    sample = test_data.sample(n=num_rows, random_state=42)
    test_data = test_data.drop(sample.index)
    train_data = pd.concat([train_data,sample]).reset_index(drop=True)
    #train_data = pd.concat([train_data,test_data],ignore_index=True)

    predictor = TabularPredictor(label=label, eval_metric='mean_absolute_error').fit(train_data, presets='best_quality', ag_args_fit={'num_gpus':1}, num_stack_levels=0,tuning_data=test_data,use_bag_holdout=True)
    x = predictor.feature_importance(test_data)

    print(x)

    submission_data = TabularDataset('B/preproc_test_estimated_B.csv')

    predictions_B = predictor.predict(submission_data)

    ###########C

    train_data = TabularDataset('C/preproc_train_observed_C.csv')
    label = 'pv_measurement'
    test_data = TabularDataset(f'C/preproc_train_estimated_C.csv')
    #train_data = pd.concat([train_data,test_data],ignore_index=True)

    percentage = 0.50
    num_rows = int(len(test_data) * percentage)
    sample = test_data.sample(n=num_rows, random_state=42)
    test_data = test_data.drop(sample.index)
    train_data = pd.concat([train_data,sample]).reset_index(drop=True)



    predictor = TabularPredictor(label=label, eval_metric='mean_absolute_error').fit(train_data, presets='best_quality', ag_args_fit={'num_gpus':1}, num_stack_levels=0,tuning_data=test_data,use_bag_holdout=True)
    x = predictor.feature_importance(test_data)

    print(x)

    submission_data = TabularDataset('C/preproc_test_estimated_C.csv')

    predictions_C = predictor.predict(submission_data)

    temp_r = pd.concat([predictions_B,predictions_C],axis=0)
    temp_r = temp_r.reset_index(drop=True)
    temp_r.index.name = 'id'
    temp_r.rename('prediction', inplace=True)
    # combined_predictions = combined_predictions.drop('Unnamed: 0', axis=1)
    temp_r.to_csv('temp_B_C.csv')

    combined_predictions = pd.concat([predictions_A,predictions_B,predictions_C], axis=0)
    combined_predictions = combined_predictions.reset_index(drop=True)
    combined_predictions.index.name = 'id'
    combined_predictions.rename('prediction', inplace=True)
    # combined_predictions = combined_predictions.drop('Unnamed: 0', axis=1)
    combined_predictions.to_csv('auto_predictions_finalv31_bare_50_80.csv')



def create_wind(df):
    pd.options.mode.chained_assignment = None
    epsilon = 1e-6

    df["windSpeed"] = np.sqrt(df["wind_speed_u_10m:ms"]**2 + df["wind_speed_v_10m:ms"]**2)
    df["windAngle"] = np.arctan2(df["wind_speed_v_10m:ms"], df["wind_speed_u_10m:ms"])

    ###
    df["global_rad"] = df["direct_rad:W"] +df["diffuse_rad:W"]

    return df


def feature_expansion(df):
    pd.options.mode.chained_assignment = None


    copy = df.copy()
    epsilon = 1e-6

    for column in copy:
        new_column_name = f'{column}_squared'
        copy[new_column_name] = copy[column] ** 2


    columns = df.columns
    column_combinations = list(combinations(columns,2))

    for combo in column_combinations:
        col1, col2 = combo
        summ = f'{col1}_{col2}_sum'
        df[summ] = df[col1] + df[col2]
        minus = f'{col1}_{col2}_minus'
        df[minus] = df[col1] - df[col2]
        mult = f'{col1}_{col2}_mult'
        df[mult] = df[col1] * df[col2]
        div= f'{col1}_{col2}_div'
        df[div] = df[col1] / df[col2].replace(0,epsilon)
    

    combined = pd.concat([copy,df], axis = 1)
    return combined
        

if __name__ == "__main__":

    pd.options.mode.chained_assignment = None

    main()









