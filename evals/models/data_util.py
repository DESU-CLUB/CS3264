def process_data(df):
    #Turn Gender to 0 and 1
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    return df