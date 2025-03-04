import os, json, tqdm, cv2, glob
from collections import defaultdict
from PIL import Image
import numpy as np

source_folder = '/media/vahid/DATA/data/animl_data/generated_video_data_processed'
cogvid_data_path = '/media/vahid/DATA/data/animl_data/cogvid_preproc_merged'

rename_paths = True
new_path = ''

def crop_image(image, target_width, target_height):
    """Crops a PIL image to the target dimensions (multiples of 8) using center crop."""
    img_width, img_height = image.size

    left = (img_width - target_width) // 2
    top = (img_height - target_height) // 2
    right = (img_width + target_width) // 2
    bottom = (img_height + target_height) // 2

    # Adjust crop region if it falls outside of the image
    left = max(0, left)
    top = max(0, top)
    right = min(img_width, right)
    bottom = min(img_height, bottom)

    return image.crop((left, top, right, bottom)), (left, top)

def get_bounding_box(image):
    """
    Calculates the bounding box of the non-transparent region of an image.
    Returns (left, upper, right, lower) or None if the image is fully transparent.
    """
    if image.mode not in ("RGBA", "LA"):
        return 0, 0, image.width, image.height
    alpha_channel = image.getchannel("A")
    bbox = alpha_channel.getbbox()
    if bbox is None:
        return None  # Fully transparent image
    return bbox


def merge_vid_with_crop(input_videos, obj_id, output_path):
    def get_frames(frames):
        # Get Maximum bounding box
        images = []
        for path in frames:
            try:
                img = Image.open(path)
                images.append(img)
            except FileNotFoundError:
                print(f"Error: Could not find image: {path}")
                continue
            except Exception as e:
                print(f"Error: Could not load image: {path}. Details: {e}")
                continue

        return images

    def crop_frames(frames):
        # Get Maximum bounding box
        max_bbox_width = 0
        max_bbox_height = 0
        for img in frames:
            bbox = get_bounding_box(img)
            if bbox:
                left, upper, right, lower = bbox
                bbox_width = right - left
                bbox_height = lower - upper
                max_bbox_width = max(max_bbox_width, bbox_width)
                max_bbox_height = max(max_bbox_height, bbox_height)

        # If any image is not transparent, assume the entire image is needed.
        if max_bbox_width == 0 or max_bbox_height == 0:
            max_bbox_width = max(img.width for img in frames) if frames else 0
            max_bbox_height = max(img.height for img in frames) if frames else 0

        ### create a margin to be sure all of the obejct is inside
        max_bbox_width *= 1.35
        max_bbox_height *= 1.35

        result = []
        for image in frames:
            cropped_img, offset = crop_image(image, max_bbox_width, max_bbox_height)
            if cropped_img.width < cropped_img.height:
                result.append(cropped_img.rotate(90, expand=True))
            else:
                result.append(cropped_img)

        return result



    vid_paths = []
    all_vids = [get_frames(v) for v in input_videos]

    for i, frames in enumerate(all_vids):
        other_clips = [v for j, v in enumerate(all_vids) if j != i]
        for ii, clip in enumerate(other_clips):
            cropped_frames = crop_frames(clip[::-1]+frames)
            merged_vid_path = f'{output_path}/{obj_id}_{i}_{ii}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(merged_vid_path, fourcc, 8, cropped_frames[0].size)
            for frame in cropped_frames:
                out.write(cv2.cvtColor(np.asarray(frame.copy().convert("RGB")), cv2.COLOR_RGB2BGR))
            out.release()
            vid_paths.append(merged_vid_path)
    return vid_paths


def merge_videos_with_reversed(input_videos, obj_id, output_path):
    """
    Merge the reverse of each video with the rest of all other videos.

    Args:
        input_videos (list): List of file paths to .mp4 videos.
        output_path (str): Path to save the final merged video.

    Returns:
        None
    """

    def rev_vid(input_path):
        # Open the input video
        cap = cv2.VideoCapture(input_path)

        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        vid_info = [fps, width, height]

        # Read frames from the video and store them in a list
        frames = []
        for _ in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        # Write frames in reverse order to the output video
        rev_frames = []
        for frame in reversed(frames):
            rev_frames.append(frame)

        return frames, rev_frames, vid_info

    if not input_videos:
        raise ValueError("Input video list is empty.")

    vid_paths = []
    all_vids = [rev_vid(v) for j, v in enumerate(input_videos)]

    for i, [vid, _, _] in enumerate(all_vids):

        other_clips = [v for j, v in enumerate(all_vids) if j != i]

        # Combine reversed clip with the rest of the clips
        for ii, clip in enumerate(other_clips):

            vid_1, rev_vid_1, [fps, width, height] = clip

            merged_vid_path = f'{output_path}/{obj_id}_{i}_{ii}.mp4'

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(merged_vid_path, fourcc, fps*2, (width, height))

            # Write frames in reverse order to the output video
            for frame in rev_vid_1:
                out.write(frame)

            for frame in vid:
                out.write(frame)

            # Release resources
            out.release()

            vid_paths.append(merged_vid_path)

    return vid_paths


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


def group_paths_by_suffix(paths):
    """
    Groups file paths based on the last substring after the '_' character in their names.

    Args:
        paths (list of str): List of file paths.

    Returns:
        dict: A dictionary where keys are the suffixes and values are lists of file paths.
    """
    groups = defaultdict(list)

    for path in paths:
        # Extract the file name from the path
        file_name = path.split('/')[-1]
        # Get the suffix after the last '_'
        suffix = file_name.split('_')[-1]
        groups[suffix].append(path)

    return dict(groups)


metadata_list = []
traj_names_rev = {'left':'right', 'right':'left', 'top':'bottom', 'bottom':'top',
                  'top-left':'bottom-right', 'top-right':'bottom-left', 'bottom-right':'top-left', 'bottom-left':'top-right'}

obj_list = os.listdir(source_folder)[:10]

vid_dir = os.path.join(cogvid_data_path, 'videos')
cond_dir = os.path.join(cogvid_data_path, 'conditions')
# os.makedirs(vid_dir, exist_ok=True)
# os.makedirs(cond_dir, exist_ok=True)
#
# for obj_name in tqdm.tqdm(obj_list):
#     obj_dir = os.path.join(source_folder, obj_name)
#     if os.path.isdir(f"{obj_dir}/video_gen_data"):
#         if os.path.isfile(f"{obj_dir}/caption_llava.json"):
#             with open(f"{obj_dir}/caption_llava.json") as f:
#                 captions = json.load(f)
#             if captions['qa'] == 'YES':
#                 traj_list = os.listdir(f"{obj_dir}/video_gen_data")
#                 traj_dict = group_paths_by_suffix(traj_list)
#                 new_vid_paths = []
#                 new_cond_vid_paths = []
#                 for k, v in traj_dict.items():
#                     input_videos = []
#                     for cnt, traj in enumerate(v):
#                         # vid_path = f"{obj_dir}/video_gen_data/{traj}/gs.mp4"
#                         vid_path = sorted(glob.glob(f"{obj_dir}/video_gen_data/{traj}/gs/*.png"))
#                         input_videos.append(vid_path)
#
#                     # new_vid_paths += merge_videos_with_reversed(input_videos, obj_name[-7:], vid_dir)
#                     new_vid_paths += merge_vid_with_crop(input_videos, obj_name[-7:], vid_dir)
#
#                     input_cond_videos = []
#                     for cnt, traj in enumerate(v):
#                         # vid_path = f"{obj_dir}/video_gen_data/{traj}/grm.mp4"
#                         vid_path = sorted(glob.glob(f"{obj_dir}/video_gen_data/{traj}/grm/*.png"))
#                         input_cond_videos.append(vid_path)
#
#
#                     # new_cond_vid_paths += merge_videos_with_reversed(input_cond_videos, obj_name[-7:], cond_dir)
#                     new_cond_vid_paths += merge_vid_with_crop(input_cond_videos, obj_name[-7:], cond_dir)
#
#                 for k, new_vid_path in enumerate(new_vid_paths):
#                     if os.path.isfile(new_vid_path):
#                         metadata_item = {}
#                         metadata_item['video'] = new_vid_path
#                         metadata_item['text'] =  captions['caption']
#                         metadata_item['condition'] = new_cond_vid_paths[k]
#                         metadata_list.append(metadata_item)

output_path = f'{cogvid_data_path}/metadata.jsonl'
# save_to_jsonl(metadata_list, output_path)

prompt_txt_path = f'{cogvid_data_path}/prompt_1k.txt'
vid_txt_path = f'{cogvid_data_path}/videos_1k.txt'
cond_txt_path = f'{cogvid_data_path}/conditions_1k.txt'

vid_list = os.listdir( os.path.join(cogvid_data_path, 'videos'))

if rename_paths:
    samples = read_jsonl(output_path)
    sel_inds = np.random.randint(len(samples), size=1_000).tolist()
    prompt_list = []
    video_list = []
    cond_list = []
    sub_prompt_list = []
    sub_video_list = []
    for n, smp in tqdm.tqdm(enumerate(samples)):
        if n not in sel_inds:
            continue
        if not os.path.isfile(smp['video']):
            print(smp['video'])
        video_list.append(smp['video'].replace(cogvid_data_path, new_path)[1:])
        cond_list.append(smp['condition'].replace(cogvid_data_path, new_path)[1:])
        prompt_list.append(smp['text'])
        # if 'sneaker' in smp['text'] or 'shoe' in smp['text'] or 'boot' in smp['text']:
        #     sub_video_list.append(video_list[-1])
        #     sub_prompt_list.append(prompt_list[-1])

    video_list = set(video_list)
    save_list_to_txt(prompt_txt_path, prompt_list)
    save_list_to_txt(vid_txt_path, video_list)
    save_list_to_txt(cond_txt_path, cond_list)
    # save_list_to_txt(sub_prompt_txt_path, sub_prompt_list)
    # save_list_to_txt(sub_vid_txt_path, sub_video_list)
