import pandas as pd
import numpy as np
from utils import normalize, calculate_score

class StudentAnalyzer:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)

    def preprocess(self):
        self.df["avg"] = self.df[["maths","science","english"]].mean(axis=1)
        self.df[["maths","science","english"]] = self.df[["maths","science","english"]].apply(normalize)

    def performance_label(self):
        self.df["performance"] = self.df["avg"].apply(
            lambda x: "Excellent" if x >= 85 else "Good" if x >= 70 else "Needs Improvement"
        )

    def ml_score(self):
        X = self.df[["maths","science","english"]].values
        weights = np.array([0.4, 0.35, 0.25])
        bias = 0.5
        self.df["ml_score"] = calculate_score(X, weights, bias)

    def get_top_students(self):
        return self.df.sort_values("ml_score", ascending=False)

