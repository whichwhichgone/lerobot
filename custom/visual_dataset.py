# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script demonstrates the use of `LeRobotDataset` class for handling and processing robotic datasets from Hugging Face.
It illustrates how to load datasets, manipulate them, and apply transformations suitable for machine learning tasks in PyTorch.

Features included in this script:
- Viewing a dataset's metadata and exploring its properties.
- Loading an existing dataset from the hub or a subset of it.
- Accessing frames by episode number.
- Using advanced dataset features like timestamp-based frame selection.
- Demonstrating compatibility with PyTorch DataLoader for batch processing.

The script ends with examples of how to batch process data using PyTorch's DataLoader.
"""

from pprint import pprint
import numpy as np
from PIL import Image
from pathlib import Path
import os

import lerobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from torchvision.transforms import v2

# Or simply explore them in your web browser directly at:
# https://huggingface.co/datasets?other=LeRobot

# Let's take this one for this example
repo_id = "u22_debug_v5_1.0.0_lerobot"
# We can have a look and fetch its metadata to know more about it:
ds_meta = LeRobotDatasetMetadata(repo_id, root="/liujinxin/dataset/bimanual/u22_debug_v5_1.0.0_lerobot")

# By instantiating just this class, you can quickly access useful information about the content and the
# structure of the dataset without downloading the actual data yet (only metadata files — which are
# lightweight).
print(f"Total number of episodes: {ds_meta.total_episodes}")
print(f"Average number of frames per episode: {ds_meta.total_frames / ds_meta.total_episodes:.3f}")
print(f"Frames per second used during data collection: {ds_meta.fps}")
print(f"Robot type: {ds_meta.robot_type}")
print(f"keys to access images from cameras: {ds_meta.camera_keys=}\n")

print("Tasks:")
print(ds_meta.tasks)
print("Features:")
pprint(ds_meta.features)

# You can also get a short summary by simply printing the object:
print(ds_meta)

# You can then load the actual dataset from the hub.
# Either load any subset of episodes:
dataset = LeRobotDataset(repo_id, root="/liujinxin/dataset/bimanual/u22_debug_v5_1.0.0_lerobot", episodes=[10, 1111, 230, 152])
debug_dir = Path("/liujinxin/zhaowei/lerobot/debugs")
debug_dir.mkdir(parents=True, exist_ok=True)

# And see how many frames you have:
print(f"Selected episodes: {dataset.episodes}")
print(f"Number of episodes selected: {dataset.num_episodes}")
print(f"Number of frames selected: {dataset.num_frames}")


# LeRobot datasets also subclasses PyTorch datasets so you can do everything you know and love from working
# with the latter, like iterating through the dataset.
# The __getitem__ iterates over the frames of the dataset. Since our datasets are also structured by
# episodes, you can access the frame indices of any episode using the episode_data_index. Here, we access
# frame indices associated to the first episode:
episode_index = 2
from_idx = dataset.episode_data_index["from"][episode_index].item()
to_idx = dataset.episode_data_index["to"][episode_index].item()

# Then we grab all the image frames from the first camera:
camera_key_0 = dataset.meta.camera_keys[0]
camera_key_1 = dataset.meta.camera_keys[1]
camera_key_2 = dataset.meta.camera_keys[2]
frames_0 = [dataset[idx][camera_key_0] for idx in range(from_idx, to_idx)]
frames_1 = [dataset[idx][camera_key_1] for idx in range(from_idx, to_idx)]
frames_2 = [dataset[idx][camera_key_2] for idx in range(from_idx, to_idx)]

test_image = frames_1[0]
test_image = test_image.permute(1, 2, 0).numpy() * 255.0
test_image = test_image.astype(np.uint8)
test_image = Image.fromarray(test_image)
test_image.save(debug_dir / f"test_image.png")


crop_transform = v2.RandomResizedCrop(size=(224, 224), scale=(0.85, 0.85))
result_image = crop_transform(test_image)
result_image.save(debug_dir / f"result_image.png")

#for i in range(len(frames_0)):
#    frames_0[i] = frames_0[i].permute(1, 2, 0).numpy() * 255.0
#    frames_0[i] = frames_0[i].astype(np.uint8)
#    frames_0[i] = Image.fromarray(frames_0[i])
#    frames_0[i].save(debug_dir / f"frames_0_{i}.png")

action = np.array([dataset[idx]["action"].numpy() for idx in range(from_idx, to_idx)])
action_file_path = os.path.join(debug_dir, f"episode_{episode_index}_action.npy")
np.save(action_file_path, action)
