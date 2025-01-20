import pandas as pd
import pickle
import os

# Load the dataset
base_dir = r'dataset'
input_csv_file = "testing_data.csv"
output_csv_file = "testing_data_output.csv"

input_csv = os.path.join(base_dir, input_csv_file)
output_csv = os.path.join(base_dir, output_csv_file)

# Load the .pkl model
modelo_path = "news_classifier.pkl"
with open(modelo_path, "rb") as file:
    modelo = pickle.load(file)

# Read the CSV file
df = pd.read_csv(input_csv, sep='\t',header=None, index_col=None)
df.columns = ["label","headline"]

# Ensure all labels are 2
df['label'] = df['label'].apply(lambda x: "2" if x != "2" else x)

# Load the model and vectorizer
with open('news_classifier.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

# Vectorize the "headline" column
headline_tfidf = loaded_vectorizer.transform(df['headline'])

# Get predictions from the model
df['predictions'] = loaded_model.predict(headline_tfidf)

# Replace the "label" column with the predictions
df['label'] = df['predictions']

# Remove the predictions column
df.drop(columns=['predictions'], inplace=True)

# Remove column names
df.columns = range(df.shape[1])

# Save the new CSV file while preserving the original format
df.to_csv(output_csv, sep='\t', index=False, header=False)
print(f"Updated file saved as {output_csv}")
