from multiprocessing import Pool, cpu_count
import os, json, tqdm, cv2, glob, shutil, zipfile
import cv2
import numpy as np
from sympy.codegen.ast import continue_

source_folder = '/media/vahid/DATA/projects/splatting_pipeline/animl_data/generated_video_data'
preproc_path = '/media/vahid/DATA/data/animl_data/generated_video_data_interpolated_processed'
caption_dir = '/media/vahid/DATA/data/animl_data/generated_video_data_processed'
cogvid_data_path = '/media/vahid/DATA/data/animl_data/cogvid_preproc_interpolate'
os.makedirs(preproc_path, exist_ok=True)


rename_paths = True
new_path = ''

def add_noise_to_frame(frame, noise_type="gaussian", amplitude=10):
    """
    Adds noise to a single frame.

    Args:
        frame (numpy.ndarray): Input frame.
        noise_type (str): Type of noise ('gaussian' or 'blur').
        amplitude (float): Amplitude of the noise.

    Returns:
        numpy.ndarray: Frame with added noise.
    """
    if noise_type == "gaussian":
        noise = np.random.normal(0, amplitude, frame.shape).astype(np.uint8)
        noisy_frame = cv2.add(frame, noise)
    elif noise_type == "blur":
        kernel_size = max(1, int(amplitude))  # Ensure kernel size is at least 1
        if kernel_size % 2 == 0:
            kernel_size += 1  # Kernel size must be odd for cv2.GaussianBlur
        noisy_frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    else:
        raise ValueError("Invalid noise_type. Choose 'gaussian' or 'blur'.")
    return noisy_frame


def process_video_with_noise(input_video_path, output_video_path, noise_type="blur", amplitude=10):
    """
    Applies noise to each frame of the video and saves the result.

    Args:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to save the output video file.
        noise_type (str): Type of noise ('gaussian' or 'blur').
        amplitude (float): Amplitude of the noise.

    Returns:
        None
    """
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {input_video_path}")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Initialize VideoWriter
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply noise
        noisy_frame = add_noise_to_frame(frame, noise_type=noise_type, amplitude=amplitude)

        # Write the noisy frame to the output video
        out.write(noisy_frame)

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
    metadata = []

    zip_path = os.path.join(source_folder, obj_name)
    obj_name = os.path.splitext(zip_path)[0].split('/')[-1]
    obj_dir = os.path.join(preproc_path, obj_name)
    caption_path = os.path.join(caption_dir, obj_name)

    if zip_path.endswith(".zip"):
        # Check if the extracted folder already exists
        if not os.path.exists(obj_dir):
            # Extract the zip file
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(obj_dir)
                # Add your custom processing code here
                print(f"extracted: {obj_name}")
            except:
                print(f"exception {obj_name}, zip extracting error!")
                # shutil.rmtree(extract_path)
        else:
            print(f"Skipping {obj_name} - already extracted")

    if os.path.isdir(f"{obj_dir}/video_gen_data"):
        if os.path.isfile(f"{caption_path}/caption_llava.json"):
            with open(f"{caption_path}/caption_llava.json") as f:
                captions = json.load(f)
            if captions['qa'] == 'YES':
                # if not ('sneaker' in captions['caption'] or 'shoe' in captions['caption'] or 'boot' in captions['caption']):
                #     return []
                shutil.copy(f"{caption_path}/caption_llava.json", f"{obj_dir}/caption_llava.json")
                traj_list = os.listdir(f"{obj_dir}/video_gen_data")
                for cnt, traj in enumerate(traj_list):
                    traj_name = traj.replace('interpolate_','')
                    caption = captions['caption']
                    vid_path = f"{obj_dir}/video_gen_data/{traj}/gs.mp4"
                    if not os.path.isfile(vid_path):
                        continue

                    new_vid_path = f"{video_dir}/{obj_name[-5:]}_{cnt:02d}_gs_{traj_name}.mp4"
                    new_vid_rev_path = f"{video_dir}/{obj_name[-5:]}_{cnt:02d}_gs_{traj_name}_rev.mp4"

                    vid_cond_path = f"{obj_dir}/video_gen_data/{traj}/grm.mp4"
                    new_vid_cond_path = f"{video_cond_dir}/{obj_name[-5:]}_{cnt:02d}_gs_{traj_name}.mp4"
                    new_vid_cond_rev_path = f"{video_cond_dir}/{obj_name[-5:]}_{cnt:02d}_gs_{traj_name}_rev.mp4"

                    if 'Camera trajectory is toward the bottom.' in caption or True:
                        if not os.path.isfile(new_vid_path):
                            shutil.copy(vid_path, new_vid_path)
                        if not os.path.isfile(new_vid_cond_path):
                            if os.path.isfile(vid_cond_path):
                                shutil.copy(vid_cond_path, new_vid_cond_path)
                            else:
                                noise_amp = np.random.randint(50, 100)
                                process_video_with_noise(new_vid_path, new_vid_cond_path, noise_type="blur", amplitude=noise_amp)

                    if os.path.isfile(new_vid_path):
                        metadata.append({
                            'video': new_vid_path.replace(cogvid_data_path, '')[1:],
                            'text': caption,
                            'condition': new_vid_cond_path.replace(cogvid_data_path, '')[1:],
                        })

                    if 'Camera trajectory is toward the bottom.' in caption or True:
                        if not os.path.isfile(new_vid_rev_path):
                            reverse_video(vid_path, new_vid_rev_path)
                        if not os.path.isfile(new_vid_cond_rev_path):
                            if os.path.isfile(vid_cond_path):
                                reverse_video(vid_cond_path, new_vid_cond_rev_path)
                            else:
                                noise_amp = np.random.randint(50, 100)
                                process_video_with_noise(new_vid_rev_path, new_vid_cond_rev_path, noise_type="blur", amplitude=noise_amp)

                    if os.path.isfile(new_vid_rev_path):
                        metadata.append({
                            'video': new_vid_rev_path.replace(cogvid_data_path, '')[1:],
                            'text': caption,
                            'condition': new_vid_cond_rev_path.replace(cogvid_data_path, '')[1:],
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
    num_job = 8
    with Pool(processes=num_job) as pool:
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
