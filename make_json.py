import os
import json
filenames = os.listdir('cats_and_dogs/train')
sample_json = []
for i in filenames:
 sample_json.append(
 {
 'image_id': i,
 'label': 'Cat' if 'cat' in i else 'Dog'
 }
 )
with open('data.json', 'w') as outfile:
 json.dump(sample_json, outfile, indent=4, sort_keys=True)