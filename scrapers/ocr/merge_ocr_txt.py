import os

def merge_txt_files(directory, output_file):
    # Open the output file in write mode (will overwrite if it exists)
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Loop over all files in the directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            print(f"Merging file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
                outfile.write("\n")

# Define the directory containing your text files and the output file path
directory_path = '../../data/MC0049597_000' 
output_filename = '.../../data/OCR1.txt'

merge_txt_files(directory_path, output_filename)
print(f"All text files have been merged into {output_filename}")