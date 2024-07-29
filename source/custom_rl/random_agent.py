# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

import omni.isaac.lab_tasks  # noqa: F401
from tasks import quadcopter
from omni.isaac.lab_tasks.utils import parse_env_cfg

from custom_network import TransformerEncoder


class vector_stacking:
    def __init__(self, max_size=10, vector_size=12):
        self.max_size = max_size
        self.vector_size = vector_size
        self.storage = []

    def add_vector(self, vector):
        # Convert vector to NumPy array or PyTorch tensor if necessary
        if isinstance(vector, list):
            vector = np.array(vector)

        # Ensure the vector is of the correct size
        assert len(vector) == self.vector_size, "Vector must be of size {}".format(self.vector_size)

        # Add the new vector to storage
        self.storage.append(vector)

        # If storage exceeds max size, remove the oldest vector
        if len(self.storage) > self.max_size:
            self.storage.pop(0)

        # Check if max size is reached and return two lists if true
        if len(self.storage) == self.max_size:
            mid_index = self.max_size // 2
            input_vector = self.storage[:mid_index]
            output_vector = self.storage[mid_index:]
            return input_vector, output_vector

        # Return None if max size is not reached
        return None, None

    def reset(self):
        self.storage = []


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()

    # create vector stacking
    obs_stack = vector_stacking(max_size=10, vector_size=12)

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # apply actions
            obs = env.step(actions)
            # print observation
            print("Full_Observation:", obs)

            input, target = obs_stack.add_vector(obs[0]['policy'][0])

            if input is not None and target is not None:
                print(f"Input: {input}")
                print(f"Target: {target}")


    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
