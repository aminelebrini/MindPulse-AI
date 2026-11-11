import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler , OneHotEncoder , RobustScaler
class DataPreparation:

    def __init__(self, data):
        self.data = pd.DataFrame(data)

    def prepare_data(self):

        x = self.data.drop(columns=['Mental_Health_Risk'])
        y = self.data['Mental_Health_Risk']
        return x, y

    def data_prepare(self):


        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])


        return num_transformer, categorical_transformer