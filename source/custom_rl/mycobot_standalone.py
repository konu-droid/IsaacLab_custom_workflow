import argparse
import torch
import numpy as np
import cv2
from time import sleep

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate a mycobot arm with camera.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sensors.camera import Camera, CameraCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from robots.mycobot import MYCOBOT_CFG


class MyCobotEnv:
    def __init__(self):
        self.setup_scene()
        self.setup_robot()
        self.setup_cameras()

    def setup_scene(self):
        self.sim = SimulationContext(
            sim_utils.SimulationCfg(device="cpu", use_gpu_pipeline=False, dt=0.005, physx=sim_utils.PhysxCfg(use_gpu=False))
        )
        self.sim.set_camera_view(eye=[1.1, 1.1, 2.0], target=[0.0, 0.0, 0.0])

        cfg = sim_utils.GroundPlaneCfg()
        cfg.func("/World/defaultGroundPlane", cfg)
        cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        cfg.func("/World/Light", cfg)
        room_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Room/simple_room.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            copy_from_source=False,
        )
        room_cfg.func("/World/Room", room_cfg, translation=(0.0, 0.0, 0.0))

        # spawn a green cubiod with colliders and rigid body
        cfg_cone_rigid = sim_utils.CuboidCfg(
            size=[0.02, 0.02, 0.02],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        )
        cfg_cone_rigid.func(
            "/World/Objects/ConeRigid", cfg_cone_rigid, translation=(0.2, 0.0, 0.1), orientation=(0.5, 0.0, 0.5, 0.0)
        )

    def setup_robot(self):
        robot_cfg = MYCOBOT_CFG
        robot_cfg.spawn.func("/World/mycobot/Robot_1", robot_cfg.spawn, translation=(0.0, 0.0, 0.05))
        self.robot = Articulation(robot_cfg.replace(prim_path="/World/mycobot/Robot.*"))

    def setup_cameras(self):
        camera_cfg = CameraCfg(
            prim_path="/World/CameraSensor",
            update_period=0,
            height=480,
            width=640,
            offset=CameraCfg.OffsetCfg(pos=(0.911, 1.252, 1.178), rot=(0.148, -0.271, -0.834, 0.456), convention="ros"),
            data_types=["rgb"],
            colorize_semantic_segmentation=False,
            colorize_instance_id_segmentation=False,
            colorize_instance_segmentation=False,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
        )
        self.camera = Camera(cfg=camera_cfg)

    def reset(self):
        self.sim.reset()
        self.sim.step()
        self.robot.update(self.sim.get_physics_dt())
        return self.get_observation()

    def get_observation(self):
        joint_states = self.robot.data.joint_pos[0].numpy()

        self.camera.update(dt=self.sim.get_physics_dt())
        image = self.camera.data.output["rgb"]
        rgb_image = None
        
        try:
            rgb_image = cv2.cvtColor(image[0].numpy(), cv2.COLOR_RGBA2RGB)
        except:
            pass
        
        return {
            "joint_states": joint_states,
            "camera_image": rgb_image
        }

    def apply_action(self, action):
        efforts = action  # torch.tensor(action).unsqueeze(0)
        self.robot.set_joint_position_target(efforts)
        self.robot.write_data_to_sim()

    def step(self):
        self.sim.step()
        self.robot.update(self.sim.get_physics_dt())
        return self.get_observation()


def main():
    env = MyCobotEnv()
    env.reset()

    print(env.robot.joint_names)

    count = 0
    while simulation_app.is_running():
        # Example: Apply a random action
        # action = np.random.uniform(-0.69, 0.69, size=len(env.robot.joint_names))

        # The joints are in order ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper_base_inner_left_joint', 'gripper_base_inner_right_joint', 'gripper_inner_finger_left_joint', 'gripper_inner_finger_right_joint', 'gripper_finger_left_joint', 'gripper_finger_right_joint']
        # So please keep the action array in this order, and the "gripper_base_inner_left_joint" "gripper_base_inner_right_joint" are the finger joints for gripping and joint1-6 are the joints for moving the arm joint1 being base joint
        action = torch.zeros_like(env.robot.data.joint_pos)
        env.apply_action(action)

        obs = env.step()
        # print(obs)

        count += 1
        if count == 1000:
            cv2.imwrite("/home/konu/Documents/IsaacLab_custom_workflow/source/custom_rl/mycobot_image_rgb.png", obs['camera_image'])
            break

    simulation_app.close()


if __name__ == "__main__":
    main()
