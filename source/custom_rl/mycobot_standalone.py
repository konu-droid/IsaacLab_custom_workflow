"""Launch Isaac Sim Simulator first."""


import argparse
import torch
from time import sleep

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate a quadcopter.")
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
from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
# from robots.franka import FRANKA_PANDA_CFG  # isort:skip
from robots.mycobot import MYCOBOT_CFG  # isort:skip


def go_circles():
    pass


def grasp(finger_left, finger_right):
    finger_left_limit = -0.69  # -40.0 degree = 0.69 radians
    finger_right_limit = 0.69
    finger_left_update = 0.0
    finger_right_update = 0.0

    if finger_left > finger_left_limit:
        finger_left_update = finger_left - 0.01

    if finger_right < finger_right_limit:
        finger_right_update = finger_right + 0.01

    return finger_left_update, finger_right_update


def main():
    """Main function."""

    # Load kit helper
    sim = SimulationContext(
        sim_utils.SimulationCfg(device="cpu", use_gpu_pipeline=False, dt=0.005, physx=sim_utils.PhysxCfg(use_gpu=False))
    )
    # Set main camera
    sim.set_camera_view(eye=[1.1, 1.1, 2.5], target=[0.0, 0.0, 0.0])

    # Spawn things into stage
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Robots
    robot_cfg = MYCOBOT_CFG
    robot_cfg.spawn.func("/World/mycobot/Robot_1", robot_cfg.spawn, translation=(0.0, 0.0, 0.05))

    # create handles for the robots
    robot = Articulation(robot_cfg.replace(prim_path="/World/mycobot/Robot.*"))

    # # Room
    # room_cfg = sim_utils.UsdFileCfg(
    #     usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Room/simple_room.usd",
    #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #         disable_gravity=False,
    #         max_depenetration_velocity=10.0,
    #         enable_gyroscopic_forces=True,
    #     ),
    #     articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    #         enabled_self_collisions=False,
    #         solver_position_iteration_count=4,
    #         solver_velocity_iteration_count=0,
    #         sleep_threshold=0.005,
    #         stabilization_threshold=0.001,
    #     ),
    #     copy_from_source=False,
    # )
    # room_cfg.func("/World/Room", room_cfg, translation=(0.0, 0.0, 0.0))

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    sleep(5)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    print(robot.find_joints("gripper_base_inner_left_joint"))
    print(robot.joint_names)
    # update buffers
    sim.step()
    robot.update(sim_dt)

    # Simulate physics
    while simulation_app.is_running():
             
        # Apply random action
        # -- generate random joint efforts
        efforts = torch.zeros_like(robot.data.joint_pos) # robot.data.joint_pos
        velocities = torch.zeros_like(robot.data.joint_pos)

        #slowing moving hte finger
        left, right = grasp(robot.data.joint_pos[0][robot.find_joints("gripper_base_inner_left_joint")[0]], robot.data.joint_pos[0][robot.find_joints("gripper_base_inner_right_joint")[0]])

        efforts[0][robot.find_joints("gripper_base_inner_left_joint")[0]] = left
        efforts[0][robot.find_joints("gripper_base_inner_right_joint")[0]] = right
        # efforts[0][robot.find_joints("panda_joint1")[0]] = 100.0
        print("pose", robot.data.joint_pos[0][robot.find_joints("gripper_base_inner_right_joint")[0]])
        print("left", left)
        print("right", right)
        print("efforts", efforts)

        # # -- apply action to the robot
        robot.set_joint_position_target(efforts)
        # robot.write_joint_state_to_sim(efforts, velocities)
        # -- write data to sim
        robot.write_data_to_sim()

        print(robot.data.joint_pos)

        # perform step
        sim.step()
        # sleep(0.2)
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        robot.update(sim_dt)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
