import pandas as pd
import torch
import numpy as np
from collections import Counter

def test_preprocessing():
    # Create a sample dataframe
    df = pd.DataFrame({
        "time": [0, 0, 1, 1, 2, 2],
        "bow": [["apple", "banana"], ["apple", "orange"], ["banana"], ["orange"], ["apple"], ["banana"]]
    })

    # Create a sample dictionary
    dictionary = {"apple": 0, "banana": 1, "orange": 2}

    # Create a sample device
    device = torch.device("cpu")

    # Create an instance of the Preprocessing class
    preprocessing = Preprocessing(df, dictionary, device)

    # Test the cs attribute
    assert preprocessing.cs == 6

    # Test the N attribute
    assert preprocessing.N == 6

    # Test the device attribute
    assert preprocessing.device == device

    # Test the unigram_logits attribute
    expected_unigram_logits = torch.tensor([-0.6931, -1.0986, -1.0986]).to(device)
    assert torch.allclose(preprocessing.unigram_logits, expected_unigram_logits)

    # Test the m_t attribute
    expected_m_t = {0: 4, 1: 2, 2: 1}
    assert preprocessing.m_t == expected_m_t

    # Test the T attribute
    assert preprocessing.T == 3

    # Test the df_idx attribute
    expected_df_idx = pd.DataFrame({
        "time": [0, 0, 1, 1],
        "bow": [["apple", "banana"], ["apple", "orange"], ["banana"], ["orange"]]
    })
    pd.testing.assert_frame_equal(preprocessing.df_idx, expected_df_idx)

test_preprocessing()