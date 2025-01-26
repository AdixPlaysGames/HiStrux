import pandas as pd

def gather_features(column_names: list, values: list, 
                    cell_id: str='sample', save_csv: bool=False, output_file: str='sample.csv'):
    """
    Processes extracted features into a pandas DataFrame for classification.
    
    Parameters:
    - column_names: list of feature names extracted from the output
    - values: list of corresponding feature values
    - cell_id: unique identifier for the cell (default: 'sample')
    - save_csv: whether to save the resulting DataFrame to a CSV file
    - output_file: name of the CSV file if saving is enabled
    """

    # Combine cell_id with feature values
    values_with_id = [cell_id] + values 
    column_names_with_id = ['cell_id'] + column_names

    features = pd.DataFrame([values_with_id], columns=column_names_with_id)
    if save_csv:
        features.to_csv(output_file, sep=';', index=False)

    return features