import pandas as pd

def save_csv_summary(input_filename, output_filename, num_rows=10):
    """
    Save a summary of a CSV file to a text file.

    Parameters:
    - input_filename: str, name of the input CSV file
    - output_filename: str, name of the output text file
    - num_rows: int, number of rows to display in the summary
    """
    df = pd.read_csv(input_filename)

    # Get the first 'num_rows' rows
    first_rows = df.head(num_rows).to_string()

    # Get column names
    columns = ", ".join(df.columns.tolist())

    # Get number of rows and columns
    total_rows, total_columns = df.shape

    # Save the summary to a text file
    with open(output_filename, 'w') as f:
        f.write(f"Total rows: {total_rows}\n")
        f.write(f"Total columns: {total_columns}\n\n")
        f.write(f"First {num_rows} rows:\n")
        f.write(first_rows)
        f.write("\n\nColumns:\n")
        f.write(columns)

    print(f"Results saved to {output_filename}")

if __name__ == "__main__":
    input_file = 'modified_botnet.csv'
    output_file = 'modified_botnet_summary.txt'

    save_csv_summary(input_file, output_file)
