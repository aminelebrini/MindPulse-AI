from src.DataExploration import DataExploration
import pandas as pd

data = pd.read_csv("data/raw/student_mental_health.csv")

data_exploration = DataExploration(data)
data_exploration.explore_data()

