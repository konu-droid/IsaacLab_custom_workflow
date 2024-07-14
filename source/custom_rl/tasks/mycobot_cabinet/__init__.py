# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka-Cabinet environment.
"""

import gymnasium as gym

from . import agents
from .mycobot_cabinet_env import MyCobotCabinetEnv, MyCobotCabinetEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="MyCobot-Cabinet-Direct-v0",
    entry_point="tasks.mycobot_cabinet:MyCobotCabinetEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MyCobotCabinetEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.FrankaCabinetPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
