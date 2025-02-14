import json

# Read the original file and filter out lines with 'cloudflare'
valid_lines = []

with open('output.jsonl', 'r') as input_file:
    for line in input_file:
        try:
            # Parse each line as JSON
            json_obj = json.loads(line)
            
            # Check if 'cloudflare' exists in any string value within the JSON object
            json_str = json.dumps(json_obj, ensure_ascii=False).lower()
            if 'cloudflare' not in json_str:
                valid_lines.append(line)
        except json.JSONDecodeError:
            # If line is not valid JSON, skip it
            continue

# Write the filtered content back to the file
with open('output.jsonl', 'w') as output_file:
    output_file.writelines(valid_lines)