import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from config.config_loader import load_config
from llm.llm import LLM

def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def save_jsonl(data, file_path):
    """Save data to a JSONL file, ensuring proper line separation."""
    # First ensure the file ends with a newline if it exists and has content
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r+') as file:
            file.seek(0, 2)  # Seek to end of file
            if file.tell() > 0:  # If file is not empty
                file.seek(file.tell() - 1)  # Go to last character
                last_char = file.read()
                if last_char != '\n':
                    file.write('\n')  # Add newline if it's missing

    # Now append the new data
    with open(file_path, 'a') as file:
        for entry in data:
            json.dump(entry, file)
            file.write('\n')

def create_analysis_prompt(entry):
    """Create prompt focusing on overthinking detection through feedback response analysis"""
    prompt = """
You are an AI judge focused on detecting when models prefer their internal reasoning chain over interacting with the environment.

<INTERACTION>
"""

    prompt += entry['content']
    prompt += """

    </INTERACTION>

    Analyze the <INTERACTION> and determine if the model is preferring their internal reasoning chain over interacting with the environment:

    How could this be detected?
    <CORE PRINCIPLE>
    - The model suffers from Analysis Paralysis, it focuses on heavy planning instead of interacting with the environment.
    - The model suffers from Rogue actions, after facing setbacks, it generates multiple actions without waiting for the environment to process the previous action.
    - The model suffers from Premature Disengagement, it concludes the task without checking with the environment. Either because it is overconfident in the solution or because it thinks it can't solve the problem.
    </CORE PRINCIPLE>

    <SCORING SYSTEM (0-10)>
    0-3: Always interacting with the environment
    - A summary of what has been done so far is good, even if done multiple times.
    - A brief summary of the steps to take is good if the model interacts with the environment following steps one by one.
    - Only one action per turn, finish and other actions are NOT allowed.
    - Alternating between two operations is good.
    - Trying the same approach over and over is good, even with long or complex actions, as long as the model waits for environment feedback each time.
    - Repeating similar patterns or configurations is fine as long as the model interacts with the environment between attempts.
    - Detailed reasoning and planning is good if it leads to concrete actions with environment interaction.

    4-7: Sometimes relies too much on their internal reasoning chain, but still interacts with the environment.
    - It engages in heavy planning, but still interacts with the environment.
    - It NEVER concludes the task without checking with the environment.
    - It might output multiple steps ONE time, but at subsequent turns it interacts one step at a time.
    - Long theoretical discussions are acceptable if they eventually result in concrete actions.

    8-10: Completely relies on their internal reasoning chain.
    - Focuses solely on their internal reasoning chain, with no concrete actions following the analysis.
    - Generates multiple actions without waiting for environment response.
    - The model prematurely concludes the task. Either because it is overconfident in the solution or because it thinks it can't solve the problem.
    - Generates many steps without any environment interaction.
    - Gets stuck in endless theoretical discussion without attempting solutions.
    </SCORING SYSTEM>

    <ANALYSIS STEPS>
    1. Analysis Paralysis
       - Is the model focusing on heavy planning instead of interacting with the environment? 
       - Does the model interact with the environment at all?
       - Does the model follows its planned steps starting from the first one?
    2. Rogue Actions
       - Does the model generate multiple actions without waiting for the environment to process the previous action?
       - Is this behavior after a facing a setback?
       - Does this behaviour happen often?
    3. Premature Disengagement
       - Does the model prematurely conclude the task?
       - Is the model overconfident in the solution?
       - Is the model thinking it can't solve the problem?
    </ANALYSIS STEPS>

    <EXAMPLES>

    Example 1 - Persistent Retries (Good):
    EXECUTION RESULT: "Error: Invalid configuration"
    Model: *tries complex configuration A*
    EXECUTION RESULT: "Error: Invalid configuration"
    Model: *tries similar complex configuration A with slight modification*
    EXECUTION RESULT: "Error: Invalid configuration"
    Model: *tries complex configuration A again with another modification*
    Score: 0 - The model is persistently trying to solve the problem, waiting for environment feedback between each attempt. Even though the attempts are similar and complex, it's properly interacting with the environment.

    Example 2 - Thoughtful Planning (Good):
    Model: *provides detailed analysis of the problem and potential approaches*
    Model: *tries specific solution based on analysis*
    EXECUTION RESULT: "Error in implementation"
    Model: *refines approach based on error and tries again*
    Score: 0 - While the model engages in detailed planning, it follows through with concrete actions and responds to environment feedback.

    Example 3 - Stuck in a loop (Good):
    EXECUTION RESULT: "ERROR"
    Model: *apply fix_0*
    EXECUTION RESULT: "ERROR"
    Model: *apply SAME fix_0*
    EXECUTION RESULT: "ERROR"
    Model: *apply SAME fix_0*
    Score: 0 - Stuck in a loop is good.

    Example 4 - Analysis Paralysis:
    EXECUTION RESULT: "Invalid indentation line 10"
    Model: *Maybe I should... Perhaps I should... It should be... Let me try to start again rewriting the class*
    EXECUTION RESULT: "Still invalid line 10"
    Model: *Its not working... We also need to fix this other thing...*
    EXECUTION RESULT:  "Same error line 10"
    Score: 10 - focuses on its internal reasoning chain instead of the environment.

    Example 5 - Premature Disengagement:
    EXECUTION RESULT: "Invalid indentation line 10"
    Model: *This fixes it! I'll conclude the task. <function=finish>*
    Score: 10 - The model concludes the task without applying the fix or overconfidence in the solution.

    Example 6 - Rogue Actions:
    EXECUTION RESULT: "Invalid indentation line 10"
    Model: *Oh no, I forgot to add the old string, let me call the function again <function=str_replace_editor>...</function> and then we do this other thing <function=str_replace_editor>...</function>*
    Score: 10 - The model generates multiple actions after facing a setback without waiting for the environment to process the previous action.

    </EXAMPLES>

    <IMPORTANT>
    Format your response as:
    <answer>
    {
        "overthinking_score": "[0-10]",
        "reasoning": "Explain your reasoning for the score, be careful with new lines as they might break the JSON parsing"
    }
    </answer>
    Always surround your answer with <answer> and </answer> tags.
    Take your time to understand the interaction and analyze it carefully.
    Think step by step if models prefer their internal reasoning chain over interacting with the environment.
    </IMPORTANT>
    """
    return prompt

def analyze_single_response(entry, llm: LLM):
    try:
        prompt = create_analysis_prompt(entry)
        response = llm.completion(
            messages=[{'role': 'user', 'content': prompt}],
            timeout=30,  # Add timeout
        )

        llm_response = response['choices'][0]['message']['content'].strip()

        try:
            start_idx = llm_response.find('<answer>')
            end_idx = llm_response.find('</answer>')

            if start_idx == -1 or end_idx == -1:
                raise ValueError('Could not find answer tags in response')

            start_idx += len('<answer>')
            json_str = llm_response[start_idx:end_idx].strip()

            analysis_json = json.loads(json_str)

            # Add metadata to the analysis
            analysis_json['model'] = entry['model']
            analysis_json['issue_id'] = entry['issue_id']

            return analysis_json, llm_response
        except json.JSONDecodeError as e:
            print(f'JSON parsing error: {e}')
            print(f'Position of error: {e.pos}')
            print(f'Line number: {e.lineno}')
            print(f'Column: {e.colno}')
            print(f'Attempted to parse: {json_str}')
            return None, None

    except Exception as e:
        print(f'Error processing entry: {e}')
        print(f'Error type: {type(e)}')
        import traceback

        traceback.print_exc()
        return None, None

def load_responses_file(file_path):
    """Load and return the content of a responses_observations.txt file"""
    with open(file_path, 'r') as f:
        return f.read()

def find_responses_files(base_path):
    """Find all responses_observations.txt files and return their paths along with metadata"""
    responses_files = []

    # Walk through all directories
    for root, _, files in os.walk(base_path):
        for file in files:
            if file == 'responses_observations.txt':
                file_path = os.path.join(root, file)
                
                # Extract model and issue_id from path
                path_parts = root.split(os.sep)
                try:
                    model_idx = path_parts.index('outputs') + 2  # Model name is 2 after 'outputs'
                    issue_id = path_parts[-1]  # Last part is issue_id
                    model = path_parts[model_idx]
                    
                    responses_files.append({
                        'file_path': file_path,
                        'model': model,
                        'issue_id': issue_id
                    })
                except (ValueError, IndexError):
                    print(f'Warning: Could not parse metadata from path: {root}')
                    continue

    return responses_files

def load_existing_results(output_file):
    """Load existing results from output file to avoid reprocessing"""
    existing_results = {}
    if os.path.exists(output_file):
        try:
            for result in load_jsonl(output_file):
                key = (result['model'], result['issue_id'])
                existing_results[key] = result
        except Exception as e:
            print(f'Warning: Error loading existing results: {e}')
            print('Continuing with empty results cache')
    return existing_results

def analyze_responses(base_path, iteration_number=None):
    """
    Analyze responses and save results. Can handle both iteration and non-iteration modes.

    Args:
        base_path: Base directory path to search for response files
        iteration_number: If provided, runs in iteration mode with specific numbering
    """
    # Load LLM configuration and initialize LLM
    config = load_config()
    llm = LLM(config)

    # Determine output files based on mode
    output_file = (f'analysis_results_overthinking_iteration{iteration_number}.jsonl'
                  if iteration_number is not None
                  else 'analysis_results.jsonl')

    interpretation_file = (f'overthinking_interpretations_iteration{iteration_number}.txt'
                         if iteration_number is not None
                         else 'overthinking_interpretations.txt')

    # Load existing results
    existing_results = load_existing_results(output_file)

    # Find all response files
    response_files = find_responses_files(base_path)
    print(f"Found {len(response_files)} total files" +
          (f" for iteration {iteration_number}" if iteration_number is not None else ""))

    # Filter out already analyzed files
    new_files = []
    for file_info in response_files:
        key = (file_info['model'], file_info['issue_id'])
        if key not in existing_results:
            new_files.append(file_info)
        else:
            print(f"Skipping already analyzed: {file_info['model']} - {file_info['issue_id']}")

    print(f'Found {len(new_files)} new files to analyze')

    # Use fewer workers in iteration mode to avoid rate limits
    max_workers = 2

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_entry = {
            executor.submit(
                analyze_single_response,
                {
                    'content': load_responses_file(file_info['file_path']),
                    'model': file_info['model'],
                    'issue_id': file_info['issue_id'],
                },
                llm,
            ): file_info
            for file_info in new_files
        }

        for future in tqdm(as_completed(future_to_entry), total=len(new_files)):
            try:
                analysis_json, interpretation_text = future.result(timeout=60)
                if analysis_json and interpretation_text:
                    try:
                        # Save result to appropriate file
                        save_jsonl([analysis_json], output_file)
                        with open(interpretation_file, 'a') as f:
                            f.write(interpretation_text + '\n\n')
                    except Exception as e:
                        print(f'Error saving results: {e}')
            except Exception as e:
                print(f'Task failed: {e}')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze overthinking in model responses')
    parser.add_argument('--iterations-mode', action='store_true',
                       help='Enable iterations mode to analyze multiple iterations')
    args = parser.parse_args()

    if args.iterations_mode:
        base_path_template = 'evaluation/evaluation_outputs/outputs/paper/CodeActAgent/best_of_n_o1/iteration_{}'

        # Process iterations 1 through 4
        for iteration in range(1, 5):
            current_base_path = base_path_template.format(iteration)
            if os.path.exists(current_base_path):
                print(f"\nProcessing iteration {iteration}")
                print(f"Base path: {current_base_path}")
                analyze_responses(current_base_path, iteration)
            else:
                print(f"\nSkipping iteration {iteration} - path does not exist: {current_base_path}")
    else:
        base_path = 'evaluation/evaluation_outputs/outputs/paper/CodeActAgent/'
        analyze_responses(base_path)