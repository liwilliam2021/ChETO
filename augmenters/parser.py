import json

data_path = './scrapers/outputs/text_output (4).json'
with open(data_path, 'r') as f:
    data = json.load(f)

for key in data.keys():
    print (key)
    print (len(data[key]))