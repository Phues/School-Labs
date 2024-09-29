import csv
import pytest

def row_to_list(s):
    return list(s.split())

# Load your dataset from the CSV file
dataset = []
with open('data/test_oil_gas_data.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        dataset.append(row)

# Test if the function correctly handles rows with missing values
# Parametrize the test function to iterate through each row in the dataset
@pytest.mark.parametrize("input_row", dataset)
def test_row_to_list_with_missing_values(input_row):
        #Complete the function to assert if any missing value is found in your input_string
        assert any([value == '?' for value in input_row]) == False, "Missing values found in input_row"