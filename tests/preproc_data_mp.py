from multiprocessing import Pool, cpu_count
import os, json, tqdm, cv2, glob, shutil

from sympy.codegen.ast import continue_

source_folder = '/media/vahid/DATA/data/animl_data/generated_video_data_processed'
cogvid_data_path = '/media/vahid/DATA/data/animl_data/cogvid_preproc_sub'


rename_paths = True
new_path = ''

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



traj_names_rev = {
    'left': 'right', 'right': 'left', 'top': 'bottom', 'bottom': 'top',
    'top-left': 'bottom-right', 'top-right': 'bottom-left',
    'bottom-right': 'top-left', 'bottom-left': 'top-right'
}


# Processing function for each object
def process_object(obj_name):
    obj_dir = os.path.join(source_folder, obj_name)
    metadata = []

    if os.path.isdir(f"{obj_dir}/video_gen_data"):
        if os.path.isfile(f"{obj_dir}/caption_llava.json"):
            with open(f"{obj_dir}/caption_llava.json") as f:
                captions = json.load(f)
            if captions['qa'] == 'YES':
                if not ('sneaker' in captions['caption'] or 'shoe' in captions['caption'] or 'boot' in captions['caption']):
                    return []
                traj_list = os.listdir(f"{obj_dir}/video_gen_data")
                for cnt, traj in enumerate(traj_list):
                    traj_name = traj.split('_')[:-1]
                    caption = captions['caption']
                    end_pos = '-'.join(traj_name)
                    vid_caption = f'{caption} Camera trajectory is toward the {end_pos}.'
                    vid_caption_reverse = f'{caption} Camera trajectory is toward the {traj_names_rev[end_pos]}.'
                    vid_path = f"{obj_dir}/video_gen_data/{traj}/gs.mp4"
                    new_vid_path = f"{video_dir}/{obj_name[-5:]}_{cnt:02d}_gs_center_to_{end_pos}.mp4"
                    new_vid_rev_path = f"{video_dir}/{obj_name[-5:]}_{cnt:02d}_gs_{traj_names_rev[end_pos]}_to_center.mp4"
                    shutil.copy(vid_path, new_vid_path)
                    reverse_video(vid_path, new_vid_rev_path)

                    vid_cond_path = f"{obj_dir}/video_gen_data/{traj}/gengs.mp4"
                    new_vid_cond_path = f"{video_cond_dir}/{obj_name[-5:]}_{cnt:02d}_gs_center_to_{end_pos}.mp4"
                    new_vid_cond_rev_path = f"{video_cond_dir}/{obj_name[-5:]}_{cnt:02d}_gs_{traj_names_rev[end_pos]}_to_center.mp4"
                    shutil.copy(vid_cond_path, new_vid_cond_path)
                    reverse_video(vid_cond_path, new_vid_cond_rev_path)
                    if os.path.isfile(new_vid_path):
                        metadata.append({
                            'video': new_vid_path.replace(cogvid_data_path,'')[1:],
                            'text': vid_caption,
                            'condition': new_vid_cond_path.replace(cogvid_data_path,'')[1:],
                        })
                    if os.path.isfile(new_vid_rev_path):
                        metadata.append({
                            'video': new_vid_rev_path.replace(cogvid_data_path,'')[1:],
                            'text': vid_caption_reverse,
                            'condition': new_vid_cond_rev_path.replace(cogvid_data_path,'')[1:],
                        })
    return metadata


# Main script
if __name__ == "__main__":
    obj_list = os.listdir(source_folder)
    metadata_list = []

    video_dir = os.path.join(cogvid_data_path, 'videos')
    video_cond_dir = os.path.join(cogvid_data_path, 'conditions')
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(video_cond_dir, exist_ok=True)

    # Use multiprocessing Pool
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm.tqdm(pool.imap(process_object, obj_list), total=len(obj_list)))

    # Flatten the list of metadata from each process
    for result in results:
        metadata_list.extend(result)

    prompt_list = []
    video_list = []
    cond_list = []
    for smp in metadata_list:
        if smp != []:
            prompt_list.append(smp['text'])
            video_list.append(smp['video'])
            cond_list.append(smp['condition'])

    prompt_txt_path = f'{cogvid_data_path}/prompt.txt'
    vid_txt_path = f'{cogvid_data_path}/videos.txt'
    cond_txt_path = f'{cogvid_data_path}/conditions.txt'

    save_list_to_txt(prompt_txt_path, prompt_list)
    save_list_to_txt(vid_txt_path, video_list)
    save_list_to_txt(cond_txt_path, cond_list)
