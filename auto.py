#find important features to not generate too much features

from autogluon.tabular import TabularDataset, TabularPredictor

import pandas as pd
from itertools import combinations

def main():
    pd.options.mode.chained_assignment = None

    train_data = TabularDataset('A/preproc_train_observed_A.csv')
    train_data.head()
    label = 'pv_measurement'
    test_data = TabularDataset(f'A/preproc_train_estimated_A.csv')

    amount_features_1 = 30
    amount_features_final = 600

    predictor = TabularPredictor(label=label, eval_metric='mean_absolute_error').fit(train_data, tuning_data=test_data)

    x = predictor.feature_importance(test_data)

    new_train = train_data[x.index[:amount_features_1]]

    expanded_train = feature_expansion(new_train)
    expanded_train['pv_measurement'] = train_data['pv_measurement']

    expanded_test = feature_expansion(test_data[x.index[:amount_features_1]])  ##.drop(columns=['pv_measurement', 'date_calc', 'date_forecast']))
    expanded_test['pv_measurement'] = test_data['pv_measurement']

    #expanded_test = pd.concat([expanded_test,temp_test], axis = 1)
    #expanded_test['pv_measurement'] = temp_test['pv_measurement']

    #expanded_test.reset_index(drop=True, inplace=True)
    expanded_train = expanded_train.loc[:, ~expanded_train.columns.duplicated(keep='first')]
    expanded_test = expanded_test.loc[:, ~expanded_test.columns.duplicated(keep='first')]

 

    #build model with new generated features

    predictor = TabularPredictor(label=label, eval_metric='mean_absolute_error').fit(expanded_train, tuning_data=expanded_test)


    y = predictor.feature_importance(expanded_test)


    final_train = expanded_train[y.index[:amount_features_final]]
    final_train['pv_measurement'] = train_data['pv_measurement']
    final_test = expanded_test[y.index[:amount_features_final]]
    final_test['pv_measurement'] = test_data['pv_measurement']

    predictor = TabularPredictor(label=label, eval_metric='mean_absolute_error').fit(final_train, tuning_data=final_test)

    submission_data = TabularDataset('A/preproc_test_estimated_A.csv')
    submission_data = feature_expansion(submission_data[x.index[:amount_features_1]])
    submission_data = submission_data.loc[:, ~submission_data.columns.duplicated(keep='first')]


    predictions_A = predictor.predict(submission_data[y.index[:amount_features_final]])

    ##########B

    train_data = TabularDataset('B/preproc_train_observed_B.csv')
    label = 'pv_measurement'
    test_data = TabularDataset(f'B/preproc_train_estimated_B.csv')


    predictor = TabularPredictor(label=label, eval_metric='mean_absolute_error').fit(train_data, presets='best_quality', tuning_data=test_data, ag_args_fit={'num_gpus':1}, num_stack_levels=0)

    submission_data = TabularDataset('B/preproc_test_estimated_B.csv')

    predictions_B = predictor.predict(submission_data)

    ###########C

    train_data = TabularDataset('C/preproc_train_observed_C.csv')
    label = 'pv_measurement'
    test_data = TabularDataset(f'C/preproc_train_estimated_C.csv')


    predictor = TabularPredictor(label=label, eval_metric='mean_absolute_error').fit(train_data, presets='best_quality', tuning_data=test_data, ag_args_fit={'num_gpus':1}, num_stack_levels=0)

    submission_data = TabularDataset('C/preproc_test_estimated_C.csv')

    predictions_C = predictor.predict(submission_data)

    combined_predictions = pd.concat([predictions_A,predictions_B,predictions_C], axis=0)
    combined_predictions = combined_predictions.reset_index(drop=True)
    combined_predictions.index.name = 'id'
    combined_predictions.rename('prediction', inplace=True)
    combined_predictions = combined_predictions.drop('Unnamed: 0', axis=1)
    combined_predictions.to_csv('auto_predictions_finalv2.csv')

if __name__ == "__main__":
    main()














def feature_expansion(df):

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
        