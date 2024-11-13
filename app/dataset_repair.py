import csv

# Specify your input and output file names
input_file = './dataset/cardio_train.csv'  # Input file with semicolon delimiters
output_file = './dataset/output.csv'  # Output file with comma delimiters

# Open the input file for reading
with open(input_file, mode='r', newline='') as infile:
    # Create a CSV reader object with semicolon as delimiter
    reader = csv.reader(infile, delimiter=';')

    # Open the output file for writing
    with open(output_file, mode='w', newline='') as outfile:
        # Create a CSV writer object with comma as delimiter
        writer = csv.writer(outfile, delimiter=',')

        # Write each row from the reader to the writer
        for row in reader:
            writer.writerow(row)

print("Delimiter successfully changed from ';' to ','")