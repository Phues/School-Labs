import csv
import pytest
import sys
sys.path.append("C:\\Users\\Benya\\SEDS_Lab4\\tp_seds4\\src\\models")
from row_2_list import row_to_list

# Load your dataset from the CSV file
dataset = []
path = 'C:\\Users\\Benya\\SEDS_Lab4\\tp_seds4\\src\\data\\house_price.csv'
with open(path, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        dataset.append(row)

# Test if the function correctly handles rows with missing values
# Parametrize the test function to iterate through each row in the dataset
#keep track of the row number using the enumerate() function

@pytest.mark.parametrize("row_number, input_row", enumerate(dataset, 1))
def test_row_to_list_with_missing_values(row_number, input_row):
    input_string = ' '.join(input_row)  # Convert list to string
    result = row_to_list(input_string, ";")  # Call your function to convert input_string to a list
    # Check if the result contains any missing values (empty elements)
    missing_values = any(value == '' for value in result)
    assert not missing_values, f"Missing values found in the row no.{row_number}: {input_row}"