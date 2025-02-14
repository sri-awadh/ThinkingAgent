import json
import os
from pathlib import Path
# Remove or comment out the HuggingFace import
# from huggingface_hub import HfApi

def convert_json_to_jsonl():
    # Create output directory if it doesn't exist
    os.makedirs("llm_completions/jsonl", exist_ok=True)
    
    # Path to base directory containing all issue directories
    base_dir = Path("./llm_completions")
    output_file = "./combined.jsonl"
    
    # Process each JSON file from all subdirectories
    with open(output_file, 'w') as outfile:
        # Iterate through all subdirectories in llm_completions
        for issue_dir in base_dir.iterdir():
            # Skip the 'jsonl' directory and non-directories
            if issue_dir.name == 'jsonl' or not issue_dir.is_dir():
                continue
                
            # Process each JSON file in the subdirectory
            for json_file in issue_dir.glob("*.json"):
                issue_name = json_file.stem  # Get filename without extension
                
                # Read JSON file
                with open(json_file, 'r') as infile:
                    data = json.load(infile)
                    # Add issue name and directory name to the data
                    data['issue_name'] = issue_name
                    data['issue_type'] = issue_dir.name
                    # Write to JSONL file
                    outfile.write(json.dumps(data) + '\n')

# Convert JSON to JSONL
convert_json_to_jsonl()

# Comment out all HuggingFace upload code
# api = HfApi()
# api.upload_large_folder(
#     folder_path=".",
#     repo_id="AlexCuadron/SWE-Bench-Verified-O1-reasoning-high-results",
#     repo_type="dataset"
# )