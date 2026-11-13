import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler , OneHotEncoder , RobustScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
class DataPreparation:

    def __init__(self, data):
        self.data = pd.DataFrame(data)

    def data_analysis_separate_x_y_data(self):

        data_frame = pd.DataFrame(self.data)

        x = self.data.drop(columns=['Mental_Health_Risk'])
        y = self.data['Mental_Health_Risk']
        return x, y

    def data_prepare(self, numerical_features, nominal_features, ordinal_features):

        data_frame = pd.DataFrame(self.data)

        numerical_columns = data_frame.select_dtypes(include=['number']).columns
        
        for column in numerical_columns:
        
            Q1 = data_frame[column].quantile(0.25)
            Q3 = data_frame[column].quantile(0.75)
            IQR = Q3 - Q1
        
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        
            outliers = data_frame[(data_frame[column] < lower_bound) | (data_frame[column] > upper_bound)]
            count_outliers = len(outliers)
        
            print(f"Column: {column}, Outliers: {count_outliers}")
        
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])

        ordinal_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(
                categories=[['Low', 'Moderate', 'High'],
                    ['Poor', 'Average', 'Good']
                            
                ],
                
            ))
        ])

        nominal_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, numerical_features),
                ('ord', ordinal_transformer, ordinal_features),
                ('nom', nominal_transformer, nominal_features)
            ]
        )


        return preprocessor