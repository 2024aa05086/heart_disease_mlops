
import numpy as np
from sklearn.linear_model import LogisticRegression

def test_model_training():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 1])

    model = LogisticRegression()
    model.fit(X, y)

    assert hasattr(model, "predict")
