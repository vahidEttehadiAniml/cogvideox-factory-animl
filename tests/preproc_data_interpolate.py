# from huggingface_hub import snapshot_download
#
# model_path = '/media/vahid/DATA/projects/CogVideo/models/CogVideoX1.5-5B-I2V'   # The local directory to save downloaded checkpoint
# snapshot_download("THUDM/CogVideoX1.5-5B-I2V", local_dir=model_path, local_dir_use_symlinks=False, repo_type='model')
# #
# # model_path = 'checkpoints/miniflux'   # The local directory to save downloaded checkpoint
# # snapshot_download("rain1011/pyramid-flow-miniflux", local_dir=model_path, local_dir_use_symlinks=False, repo_type='model')

import os, json, tqdm, cv2, glob
import shutil

source_folder = '/media/vahid/DATA/projects/splatting_pipeline/animl_data/generated_video_data'
cogvid_data_path = '/media/vahid/DATA/data/animl_data/cogvid_preproc_interpolate_'


rename_paths = True
new_path = ''

def read_jsonl(file_path):
    """
    Reads a JSONL file and returns a list of dictionaries.

    Parameters:
        file_path (str): The path to the JSONL file.

    Returns:
        list: A list of dictionaries read from the JSONL file.
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_to_jsonl(data, output_path):
    """
    Saves a list of dictionaries to a JSONL file.

    Parameters:
        data (list): A list of dictionaries to be saved.
        output_path (str): The path to the output JSONL file.
    """
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def reverse_video(input_path, output_path):
    # Open the input video
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Read frames from the video and store them in a list
    frames = []
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    # Write frames in reverse order to the output video
    for frame in reversed(frames):
        out.write(frame)

    # Release resources
    cap.release()
    out.release()


def save_list_to_txt(file_path, string_list):
    """
    Save a list of strings to a .txt file, with each string on a new line.

    Parameters:
    - file_path (str): The path to the file where the list will be saved.
    - string_list (list of str): The list of strings to save.
    """
    with open(file_path, 'w') as file:
        for line in string_list:
            file.write(line + '\n')


def load_list_from_txt(file_path):
    """
    Load a list of strings from a .txt file, with each line as a separate string in the list.

    Parameters:
    - file_path (str): The path to the file to load the list from.

    Returns:
    - list of str: A list where each item is a line from the file.
    """
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]





metadata_list = []
traj_names_rev = {'left':'right', 'right':'left', 'top':'bottom', 'bottom':'top',
                  'top-left':'bottom-right', 'top-right':'bottom-left', 'bottom-right':'top-left', 'bottom-left':'top-right'}


obj_list = os.listdir(source_folder)

for obj_name in tqdm.tqdm(obj_list):
    new_obj_dir = os.path.join(cogvid_data_path, obj_name)
    obj_dir = os.path.join(source_folder, obj_name)
    if os.path.isdir(f"{obj_dir}/video_gen_data"):
        if os.path.isfile(f"{obj_dir}/caption_llava.json"):
            with open(f"{obj_dir}/caption_llava.json") as f:
                captions = json.load(f)
            if captions['qa'] == 'YES':
                os.makedirs(new_obj_dir, exist_ok=True)
                traj_list = os.listdir(f"{obj_dir}/video_gen_data")
                for cnt, traj in enumerate(traj_list):
                    traj_name = traj.split('_')[:-1]
                    caption = captions['caption']
                    end_pos = f'{traj_name[0]}'
                    for nm in traj_name[1:]:
                        end_pos += f'-{nm}'
                    vid_caption = f'{caption} Camera trajectory is toward the {end_pos}.'
                    vid_caption_reverse = f'{caption} Camera trajectory is toward the {traj_names_rev[end_pos]}.'
                    vid_path = f"{obj_dir}/video_gen_data/{traj}/gs.mp4"
                    new_vid_path = f"{new_obj_dir}/{cnt:02d}_gs_center_to_{end_pos}.mp4"
                    new_vid_rev_path = f"{new_obj_dir}/{cnt:02d}_gs_{traj_names_rev[end_pos]}_to_center.mp4"
                    if 'Camera trajectory is toward the bottom.' in vid_caption or True:
                        shutil.copy(vid_path, new_vid_path)
                        if os.path.isfile(new_vid_path):
                            metadata_item = {}
                            metadata_item['video'] = new_vid_path
                            metadata_item['text'] = vid_caption
                            metadata_item['latent'] = new_vid_path.replace('.mp4','_latent_384p.pt')
                            metadata_item['text_fea'] = new_vid_path.replace('.mp4','_feat_384p.pt')
                            metadata_list.append(metadata_item)
                    if 'Camera trajectory is toward the bottom.' in vid_caption_reverse or True:
                        reverse_video(vid_path, new_vid_rev_path)
                        if os.path.isfile(new_vid_rev_path):
                            metadata_item_rev = {}
                            metadata_item_rev['video'] = new_vid_rev_path
                            metadata_item_rev['text'] = vid_caption_reverse
                            metadata_item_rev['latent'] = new_vid_rev_path.replace('.mp4','_latent_384p.pt')
                            metadata_item_rev['text_fea'] = new_vid_rev_path.replace('.mp4','_feat_384p.pt')
                            metadata_list.append(metadata_item_rev)

output_path = f'{cogvid_data_path}/metadata.jsonl'
save_to_jsonl(metadata_list, output_path)

prompt_txt_path = f'{cogvid_data_path}/prompt.txt'
vid_txt_path = f'{cogvid_data_path}/videos.txt'
sub_prompt_txt_path = f'{cogvid_data_path}/sub_prompt.txt'
sub_vid_txt_path = f'{cogvid_data_path}/sub_videos.txt'

if rename_paths:
    samples = read_jsonl(output_path)
    prompt_list = []
    video_list = []
    sub_prompt_list = []
    sub_video_list = []
    for smp in tqdm.tqdm(samples):
        video_list.append(smp['video'].replace(cogvid_data_path, new_path)[1:])
        prompt_list.append(smp['text'])
        # if 'sneaker' in smp['text'] or 'shoe' in smp['text'] or 'boot' in smp['text']:
        #     sub_video_list.append(video_list[-1])
        #     sub_prompt_list.append(prompt_list[-1])


    save_list_to_txt(prompt_txt_path, prompt_list)
    save_list_to_txt(vid_txt_path, video_list)
    # save_list_to_txt(sub_prompt_txt_path, sub_prompt_list)
    # save_list_to_txt(sub_vid_txt_path, sub_video_list)
