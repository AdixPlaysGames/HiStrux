import numpy as np
import pandas as pd

def calculate_f_trans(hic_df):
    trans_interactions = hic_df[hic_df['chromosome_1'] != hic_df['chromosome_2']]
    total_interactions = len(hic_df)
    trans_count = len(trans_interactions)
    f_trans = trans_count / total_interactions if total_interactions > 0 else 0
    return f_trans


def calculate_mean_contact_length(hic_df):
    return (np.mean(abs(hic_df['end_1'] - hic_df['start_1'])) +
            np.mean(abs(hic_df['end_2'] - hic_df['start_2']))) / 2


def calculate_std_contact_length(hic_df):
    return (np.std(abs(hic_df['end_1'] - hic_df['start_1'])) +
            np.std(abs(hic_df['end_2'] - hic_df['start_2']))) / 2


def compute_basic_metrics(hic_df: pd.DataFrame):
    """
    Computes basic Hi-C interaction metrics: f_trans, mean contact length, and standard deviation of contact lengths.
    
    Parameters:
    -----------
    hic_df : pd.DataFrame
        DataFrame containing Hi-C interaction data.
    
    Returns:
    --------
    dict
        A dictionary with the following keys:
        - 'f_trans': Fraction of trans interactions.
        - 'mean_contact_length': Mean contact length.
        - 'std_contact_length': Standard deviation of contact lengths.
    """
    f_trans = calculate_f_trans(hic_df)
    mean_contact_length = calculate_mean_contact_length(hic_df)
    std_contact_length = calculate_std_contact_length(hic_df)

    return {
        'f_trans': f_trans,
        'mean_contact_length': mean_contact_length,
        'std_contact_length': std_contact_length
    }