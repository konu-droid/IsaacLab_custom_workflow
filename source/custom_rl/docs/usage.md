# MyCobotEnv Documentation

## Install

Follow instructions in [docker.rst](docs/source/deployment/docker.rst)

## Usage

1. Once you start the container using `./docker/container.sh enter`, use the command `isaaclab -p source/workspace/mycobot_standalone.py` to launch the simulation.

2. Any modification to the code can also be made in `mycobot_standalone.py` where you can use `apply_actions` to send actions to the robot and `get_observation` or `step` function to get the observations of joint and camera.

3. Observations are given out in a dict format like such:

   ```python
   {
       "joint_states": joint_states,
       "camera_image": rgb_image
   }
   ```
4. The actions need to be send as an array of shape `[1, 12]` where 1 is the for batch size and 12 variables should be in the order 
    ```python
    ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper_base_inner_left_joint', 'gripper_base_inner_right_joint', 'gripper_inner_finger_left_joint', 'gripper_inner_finger_right_joint', 'gripper_finger_left_joint', 'gripper_finger_right_joint']
    ```
5. Here `gripper_base_inner_left_joint` and `gripper_base_inner_right_joint` are the finger joints and these need to move to grasp object. Joint1-6 are the joints from base link to the wrist joint.
6. the limits for these joints are as follows (radians):
    | Joint | Lower Limit | Upper Limit |
    |-------|-------------|-------------|
    | gripper_base_inner_left_joint | -0.69 | 0.14 |
    | gripper_base_inner_right_joint | -0.14 | 0.69 |
    | joint6 | -3.0 | 3.0 |
    | joint5 | -2.8 | 2.8 |
    | joint4 | -2.8 | 2.8 |
    | joint3 | -2.8 | 2.8 |
    | joint2 | -2.8 | 2.8 |
    | joint1 | -2.8 | 2.8 |

7. New eniversonment or elements to environments can be added to `setup_scene()` fucnction in `MyCobotEnv` class, take `room_cfg` as example