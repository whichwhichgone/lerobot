from pathlib import Path
from PIL import Image
from flask import Flask, jsonify, request
import argparse

import numpy as np
import json
import torch

from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy


VISION_IMAGE_SIZE = 224


class VLAServer:
    def __init__(self, args):
        # Create a directory to store the video of the evaluation
        self.output_directory = Path(args.output_path)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        # Load the policy
        self.policy_path = Path(args.policy_path)
        self.policy = PI0Policy.from_pretrained(self.policy_path)
        self.policy.reset()

        # Select your device
        self.device = "cuda"

    def compose_input(
        self, head_img, left_img, right_img, instruction, state, debug=True
    ):
        head_img = Image.fromarray(head_img)
        left_img = Image.fromarray(left_img)
        right_img = Image.fromarray(right_img)

        if debug:
            # images for final input
            head_img.save(self.output_directory / "eval_head_img.png")
            left_img.save(self.output_directory / "eval_left_img.png")
            right_img.save(self.output_directory / "eval_right_img.png")

        head_img = (
            torch.from_numpy(np.asarray(head_img) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        left_img = (
            torch.from_numpy(np.asarray(left_img) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        right_img = (
            torch.from_numpy(np.asarray(right_img) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        state = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)

        # images have the shape of (B, C, H, W), and the type is torch.float32
        # state has the shape of (B, 16), and the type is torch.float32
        # keep the order of the input features as the same as the original policy
        input_features = {
            "observation.images.hand_image_left": left_img,
            "observation.images.image": head_img,
            "observation.images.hand_image_right": right_img,
            "observation.state": state,
            "task": [instruction],
        }
        return input_features

    def generate_action(self, batch):

        # Predict the next action with respect to the current observation
        with torch.inference_mode():
            self.policy.to(self.device)
            action_chunk = []
            for _ in range(self.policy.config.n_action_steps):
                action = self.policy.select_action(batch)
                action_chunk.append(action)
            action_chunk = torch.cat(action_chunk, dim=0)

        # Prepare the action for the environment
        numpy_action = action_chunk.to("cpu").numpy()

        # np.ndarray and its shape is (T, 16)
        return numpy_action


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy-path",
        type=str,
        default="/liujinxin/zhaowei/lerobot/outputs/train/2025-07-17/01-20-31_debug_u22_v5_no_state/checkpoints/last/pretrained_model",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/liujinxin/zhaowei/lerobot/outputs/eval/pi0_server",
    )
    parser.add_argument(
        "--port", type=int, default=9002, help="Port number for flask server"
    )
    args = parser.parse_args()

    # Start the server (Flask)
    flask_app = Flask(__name__)
    vla_robot = VLAServer(args)

    # Define the route for remote requests
    @flask_app.route("/predict", methods=["POST"])
    def predict():
        if request.method == "POST":
            head_img = np.frombuffer(request.files["head_img"].read(), dtype=np.uint8)
            head_img = head_img.reshape((VISION_IMAGE_SIZE, VISION_IMAGE_SIZE, 3))
            left_img = np.frombuffer(request.files["left_img"].read(), dtype=np.uint8)
            left_img = left_img.reshape((VISION_IMAGE_SIZE, VISION_IMAGE_SIZE, 3))
            right_img = np.frombuffer(request.files["right_img"].read(), dtype=np.uint8)
            right_img = right_img.reshape((VISION_IMAGE_SIZE, VISION_IMAGE_SIZE, 3))

            # instructions and robot_obs for final input
            content = request.files["json"].read()
            content = json.loads(content)
            instruction = content["instruction"]
            state = content["state"]

            # compose the input
            batch = vla_robot.compose_input(
                head_img, left_img, right_img, instruction, state
            )
            action = vla_robot.generate_action(batch)
            return jsonify(action.tolist())

    # Run the server
    flask_app.run(host="0.0.0.0", port=args.port)
