import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

predictions_A = TabularDataset('auto_predictions_final_A.csv')
predictions_B = TabularDataset('auto_predictions_final_B.csv')
predictions_C = TabularDataset('auto_predictions_final_C.csv')


combined_predictions = pd.concat([predictions_A,predictions_B,predictions_C], axis=0)
combined_predictions = combined_predictions.reset_index(drop=True)
combined_predictions.index.name = 'id'
print(combined_predictions)
combined_predictions.rename(columns={combined_predictions.columns[1]: 'prediction'}, inplace=True)
combined_predictions = combined_predictions.drop('Unnamed: 0', axis=1)

combined_predictions.to_csv('auto_predictions_finalv2.csv')