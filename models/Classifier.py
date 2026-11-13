from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV , cross_validate , StratifiedKFold
import pandas as pd

class Classifier:
    def __init__(self, preprocessor):
        self.models = {
            "Logistic Regression": LogisticRegression(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "SVC": SVC(probability=True, random_state=42),
            "Gradient Boosting" : GradientBoostingClassifier(random_state=42)
        }
        self.preprocessor = preprocessor
        self.result = []
        self.best_pipline = None
        self.best_model_name = None

    def train_predict_model(self, X_train, Y_train, X_test, Y_test):

        f1_best = -1.0
        
        for model_name, classifier_model in self.models.items():
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", self.preprocessor),
                    ("classifier", classifier_model)
                ]
            )

            pipeline.fit(X_train, Y_train)


            Y_predict = pipeline.predict(X_test)

            Y_probabilite = pipeline.predict_proba(X_test)

            accuracy = accuracy_score(Y_test, Y_predict)
            precision = precision_score(Y_test, Y_predict, average='macro')
            recall = recall_score(Y_test, Y_predict, average='macro')
            f1score = f1_score(Y_test, Y_predict, average='macro')
            matrix_confusion = confusion_matrix(Y_test, Y_predict)
            roc_and_auc_score = roc_auc_score(Y_test, Y_probabilite , average='macro', multi_class='ovr')

            self.result.append(
                {
                    "name" : model_name,
                    "accuracy" : accuracy,
                    "precision" : precision,
                    "recall" : recall,
                    "f1_score" : f1score,
                    "matrix_confusion" : matrix_confusion,
                    "roc_and_auc_score" : roc_and_auc_score
                }
            )

            if f1score > f1_best:
                f1_best = f1score
                self.best_pipline = pipeline
                self.best_model_name = model_name

        result_df = pd.DataFrame(self.result)

        return result_df

    def optimisation_evaluation(self, params_grid, X_train, Y_train, X_test, Y_test):

        stratified_kfold = StratifiedKFold(
            n_splits=5, 
            shuffle=True, 
            random_state=42
            )
        best_result = []
        best_cv_score = -1.0
        for name, model in self.models.items():
            pipleline = Pipeline(
                    steps=[
                        ("preprocessor", self.preprocessor),
                        ("classifier", model)
                    ]
                )
            
            GridSearchcv = GridSearchCV(
                estimator=pipleline,
                param_grid=params_grid[name],
                cv=stratified_kfold,
                scoring='f1_macro',
                n_jobs=-1
            )
            
            GridSearchcv.fit(X_train, Y_train)

            y_perdict = GridSearchcv.best_estimator_.predict(X_test)
            f1_score_macro = f1_score(Y_test, y_perdict, average='macro')
            best_result.append({
                "model_name" : name,
                "best_params" : GridSearchcv.best_params_,
                "best_cv_f1_score": GridSearchcv.best_score_,
                "best_f1_score_test" : f1_score_macro,
                "best_estimator" : GridSearchcv.best_estimator_,
            })

            if GridSearchcv.best_score_ > best_cv_score:
                best_cv_score = GridSearchcv.best_score_
                self.best_pipline = GridSearchcv.best_estimator_
                self.best_model_name = name

        return best_result









    