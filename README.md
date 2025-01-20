# News Classification Script

This Python script processes a dataset of news headlines and classifies them using a pre-trained machine learning model. The script ensures proper formatting of the dataset and generates an updated CSV file with the new classifications.

## Features

- Reads a dataset of news headlines in tab-separated format.
- Ensures all labels in the dataset are of a specific value (`2` in this case).
- Utilizes a pre-trained model (`news_classifier.pkl`) and a vectorizer (`tfidf_vectorizer.pkl`) to classify the headlines.
- Outputs an updated dataset with the predicted labels.

## Requirements

- Python 3.8 or higher
- Required Python libraries:
  - pandas
  - pickle
- Pre-trained files:
  - `news_classifier.pkl`
  - `tfidf_vectorizer.pkl`

## Directory Structure

```
project/
│
├── dataset/
│   ├── testing_data.csv       # Input dataset (tab-separated)
│   ├── testing_data_output.csv # Output dataset
│
├── news_classifier.pkl        # Pre-trained classification model
├── tfidf_vectorizer.pkl       # Pre-trained TF-IDF vectorizer
├── script.py                  # Main script
```

## Usage

1. Place the input dataset (`testing_data.csv`) in the `dataset` directory.
2. Ensure the pre-trained model (`news_classifier.pkl`) and vectorizer (`tfidf_vectorizer.pkl`) are in the root directory.
3. Run the script:
   ```bash
   python script.py
   ```
4. The script will process the dataset and save the updated file as `testing_data_output.csv` in the `dataset` directory.

## Input Format

The input CSV file must:
- Use tab (`\t`) as the delimiter.
- Contain two columns: `label` and `headline`.
- Exclude a header row.

Example:
```
1   Example headline 1
0   Example headline 2
```

## Output Format

The output CSV file:
- Retains the tab-separated format.
- Replaces the `label` column with the predicted values.
- Does not include column headers.

Example:
```
2   Example headline 1
2   Example headline 2
```

## Notes

- Ensure that the input file format adheres to the specified structure.
- Modify the `base_dir`, `input_csv_file`, and `output_csv_file` variables in the script if your directory structure is different.
- The script assumes the presence of a valid pre-trained model and vectorizer.

## Contact

Álvaro Ruedas Mora
    
- Data Scientist from **Ironhack**
- Industrial and Automation Electronics Engineer from **Polytechnic University of Madrid**
- MBA from **EAE Business School**
