import pandas as pd
import unittest

def filter_words(df, bow_col, dictionary):
    bow_filtered = df[bow_col].apply(
        lambda x: list(
            filter(lambda x: x is not None, [dictionary.get(w, None) for w in x])
        )
    )
    return bow_filtered

'''
class TestFilterWords(unittest.TestCase):
    def test_filter_words(self):
        df = pd.DataFrame({
            'bow_col': [['apple', 'banana', 'cherry'], ['orange', 'banana', 'grape'], ['apple', 'kiwi']]
        })
        dictionary = {'apple': 1, 'banana': 2, 'cherry': 3, 'grape': 4}
        result = filter_words(df, 'bow_col', dictionary)
        expected = pd.Series([
            [1, 2, 3],
            [2, 4],
            [1]
        ], name='bow_col')
        pd.testing.assert_series_equal(result, expected)

if __name__ == '__main__':
    unittest.main()
'''
df = pd.DataFrame({
    'bow_col': [['apple', 'banana', 'cherry'], ['orange', 'banana', 'grape'], ['apple', 'kiwi']]
    })
dictionary = {'apple': 1, 'banana': 2, 'cherry': 3, 'grape': 4}
result = filter_words(df, 'bow_col', dictionary)
expected = pd.Series([
            [1, 2, 3],
            [2, 4],
            [1]
], name='bow_col')
print(result)