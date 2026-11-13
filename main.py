from src.DataExploration import DataExploration
from src.DataPreparation import DataPreparation
from models.Classifier import Classifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import os
data = pd.read_csv("data/raw/student_mental_health.csv")

data_exploration = DataExploration(data)
data_exploration.explore_data()

data_preparation = DataPreparation(data)
x, y = data_preparation.data_analysis_separate_x_y_data()

X_train, X_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
    )

print("\nTraining set shape:", X_train.shape)
print("\nTrain length:", len(X_train))
print("\nTest set shape:", X_test.shape)
print("\nTest length:", len(X_test))

X_train.drop(columns=["Age"])
numerical_features = X_train.select_dtypes(include=['number']).columns.tolist()
nominal_features = ["Part_Time_Job", "Previous_Mental_Health_History"]
ordinal_features = ["Physical_Activity","Diet_Quality"]

preprocessor = data_preparation.data_prepare(numerical_features, nominal_features, ordinal_features)


classifier = Classifier(preprocessor)

risk_mapping = {
            'Low': 0,
            'Medium': 1,
            'High': 2
        }

Y_train_maped = y_train.map(risk_mapping)
Y_test_maped = y_test.map(risk_mapping)

result = classifier.train_predict_model(X_train, Y_train_maped, X_test, Y_test_maped)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(result)

matrice_confusion = result.loc[0, "matrix_confusion"]


plt.figure(figsize=(8, 6))
sns.heatmap(matrice_confusion, 
            annot=True,
            cmap='Blues',
            fmt='d',
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'],
            cbar=True
            )

plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Labels', fontsize=12)
plt.ylabel('Real Labels', fontsize=12)
plt.tight_layout()
plt.show()

print("evaluation and optimization of models !")

params_grid = {
    "Logistic Regression" : {
        'classifier__penalty' : ['l1', 'l2'],
        'classifier__solver': ['saga'],
        'classifier__C': [0.01, 0.1, 1.0, 10.0],
        'classifier__max_iter' : [1000]
    }, 
    "Random Forest" : {
        "classifier__n_estimators" : [100, 200],
        "classifier__max_depth" : [10, 20],
        "classifier__min_samples_split" : [2, 5]
    },
    "SVC" : {
        "classifier__C" : [0.01, 0.1, 1.0, 10.0],
        "classifier__max_iter" : [5000],
        "classifier__kernel" : ['linear']
    },
    "Gradient Boosting" : {
        "classifier__n_estimators" : [100, 200],
        "classifier__learning_rate" : [0.01, 0.1, 0.2],
        "classifier__max_depth" : [3, 5]
    }
}


result = classifier.optimisation_evaluation(params_grid,X_train, Y_train_maped, X_test, Y_test_maped)

result_df = pd.DataFrame(result)
pd.set_option('display.max_columns', None)      
pd.set_option('display.max_colwidth', None) 
pd.set_option('display.width', 1000)
result_df.sort_values(by="best_f1_score_test", ascending=False).reset_index(drop=True)
print(result_df)

print("best model!")
best_model = {
    "model_name" : classifier.best_model_name,
    "best_pipline" : classifier.best_pipline
}
result_best_model_df = pd.DataFrame([best_model])
# pd.set_option('display.max_columns', None)      
# pd.set_option('display.max_colwidth', None) 
# pd.set_option('display.width', 1000)
# print(result_best_model_df)


model_file_name = f"models/best_model/{best_model['model_name']}.joblib"
save_path = os.path.dirname(model_file_name)

os.makedirs(save_path, exist_ok=True)

joblib.dump(classifier.best_pipline, model_file_name)
print(f"Success to save best model in: {model_file_name}")



