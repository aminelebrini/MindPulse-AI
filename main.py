from src.DataExploration import DataExploration
from src.DataPreparation import DataPreparation
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

import pandas as pd

data = pd.read_csv("data/raw/student_mental_health.csv")

data_exploration = DataExploration(data)
data_exploration.explore_data()

data_preparation = DataPreparation(data)
x, y = data_preparation.prepare_data()

X_train, X_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.2, 
    random_state=42
    )

print("Training set shape:", X_train.shape)
print("Train length:", len(X_train))
print("Test set shape:", X_test.shape)
print("Test length:", len(X_test))

numerical_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

num_transformer, categorical_transformer = data_preparation.data_prepare()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)