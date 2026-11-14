"""Streamlit dashboard for MindPulse predictions and model information."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_mental_health.csv"
MODEL_DIRECTORY = PROJECT_ROOT / "models" / "best_model"
TARGET_COLUMN = "Mental_Health_Risk"
RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the training data used to define the prediction form."""
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    """Load the serialized sklearn pipeline once per Streamlit process."""
    model_files = list(MODEL_DIRECTORY.glob("*.joblib"))
    if not model_files:
        raise FileNotFoundError(
            f"Aucun modèle trouvé dans {MODEL_DIRECTORY}. "
            "Lancez le DAG ou main.py d'abord."
        )
    model_path = max(model_files, key=lambda path: path.stat().st_mtime)
    return joblib.load(model_path), model_path


def render_prediction_page(data: pd.DataFrame, model) -> None:
    st.header("Prédiction du risque de santé mentale")
    st.write("Renseignez les informations de l'étudiant puis lancez la prédiction.")

    with st.form("prediction_form"):
        values = {}
        columns = [column for column in data.columns if column != TARGET_COLUMN]
        for column in columns:
            series = data[column]
            if not is_numeric_dtype(series):
                options = sorted(series.dropna().unique().tolist())
                values[column] = st.selectbox(column.replace("_", " "), options)
            else:
                default = float(series.median())
                values[column] = st.number_input(
                    column.replace("_", " "),
                    value=default,
                    min_value=float(series.min()),
                    max_value=float(series.max()),
                )

        submitted = st.form_submit_button("Prédire", type="primary")

    if submitted:
        input_data = pd.DataFrame([values], columns=columns)
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        predicted_label = RISK_LABELS.get(int(prediction), str(prediction))

        st.success(f"Niveau de risque prédit : **{predicted_label}**")
        probability_data = pd.DataFrame(
            {"Niveau": [RISK_LABELS.get(index, str(index)) for index in model.classes_],
             "Probabilité": probabilities}
        )
        st.bar_chart(probability_data.set_index("Niveau"))


def render_model_page(data: pd.DataFrame, model) -> None:
    st.header("Informations sur le modèle")
    st.write("Détails du pipeline sauvegardé utilisé par l'application.")

    classifier = model.named_steps.get("classifier")
    preprocessor = model.named_steps.get("preprocessor")
    col1, col2, col3 = st.columns(3)
    model_path = getattr(model, "_mindpulse_path", MODEL_DIRECTORY / "modèle")
    col1.metric("Modèle", model_path.stem)
    col2.metric("Observations d'entraînement", len(data))
    col3.metric("Variables", len(data.columns) - 1)

    st.subheader("Pipeline")
    st.code(repr(model), language="text")

    if classifier is not None:
        st.write(f"**Estimateur :** `{classifier.__class__.__name__}`")
        if hasattr(classifier, "classes_"):
            st.write("**Classes :**", [RISK_LABELS.get(int(c), str(c)) for c in classifier.classes_])

    if preprocessor is not None:
        st.write("**Prétraitements :**")
        st.json(
            {
                name: list(columns) if not isinstance(columns, str) else columns
                for name, _, columns in preprocessor.transformers
            }
        )

    st.subheader("Colonnes attendues")
    st.dataframe(pd.DataFrame({"Colonne": data.columns}), hide_index=True)


def main() -> None:
    st.set_page_config(page_title="MindPulse AI", page_icon="🧠", layout="wide")
    st.title("MindPulse AI")
    st.caption("Évaluation du risque de santé mentale chez les étudiants")

    try:
        data = load_data()
        model, model_path = load_model()
        model._mindpulse_path = model_path
    except (FileNotFoundError, OSError, ValueError) as error:
        st.error(str(error))
        st.stop()

    page = st.sidebar.radio("Navigation", ["Prédiction", "Informations du modèle"])
    if page == "Prédiction":
        render_prediction_page(data, model)
    else:
        render_model_page(data, model)


if __name__ == "__main__":
    main()
