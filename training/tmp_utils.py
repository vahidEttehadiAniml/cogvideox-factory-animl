import json
import os


def process_jsonl_file(input_file, output_dir='output'):
    """
    Read a JSONL file and extract prompts and video paths into separate text files.

    Parameters:
    -----------
    input_file : str
        Path to the input JSONL file
    output_dir : str, optional
        Directory to save output files (default is 'output')

    Returns:
    --------
    None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Paths for output files
    prompts_file = os.path.join(output_dir, 'prompts.txt')
    videos_file = os.path.join(output_dir, 'videos.txt')

    # Open files for writing
    with open(input_file, 'r') as jsonl_file, \
            open(prompts_file, 'w') as prompts_out, \
            open(videos_file, 'w') as videos_out:

        # Enumerate to keep track of line numbers
        for line_num, line in enumerate(jsonl_file, 1):
            try:
                # Parse JSON line
                data = json.loads(line)

                # Extract prompt and video path
                prompt = data.get('prompt', '')
                video_path = data.get('video', '')

                # Write to respective files
                prompts_out.write(f"{prompt}\n")
                videos_out.write(f"{video_path}\n")

            except json.JSONDecodeError:
                print(f"Error decoding JSON on line {line_num}. Skipping.")
            except Exception as e:
                print(f"Unexpected error processing line {line_num}: {e}")

    print(f"Processing complete. Files saved in {output_dir}:")
    print(f"- {prompts_file}")
    print(f"- {videos_file}")


# Example usage
if __name__ == "__main__":
    output_dir = "/mnt/data/cogvid_preproc_sub_latents"
    process_jsonl_file(f'{output_dir}/data.jsonl',output_dir=output_dir)