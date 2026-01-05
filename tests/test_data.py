
import pandas as pd

def test_dataset_loads():
    df = pd.read_csv("data/heart.csv")
    assert df.shape[0] > 0
