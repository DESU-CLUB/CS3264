# Outputs 30 random rows of whatever csv for proof of concept

import pandas as pd

def trim_csv(filepath, filename):
    # Select 30 random rows
    df = pd.read_csv(filepath)
    state = 41
    random_rows = df.sample(n=30, random_state=state)
    random_rows.to_csv(filename +"_trimmed_rs" + str(state) + ".csv", index=False)

trim_csv("diabetes.csv", "diabetes")