import json
import pandas as pd

# Assuming the JSON is a list of dictionaries
path_ = 'outputs/OpenAI_predictions_output.json'
output_path = path_.replace('.json', '.csv')
with open(path_, 'r') as f:
    data = json.load(f)
# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv(output_path, index=False)

print("CSV file 'qa_output.csv' has been created.")
