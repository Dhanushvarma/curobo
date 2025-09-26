#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

try:
    # Third Party
    import isaacsim
except ImportError:
    pass


# Third Party
import cv2
import torch

a = torch.zeros(4, device="cuda:0")
# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "width": "1920",
        "height": "1080",
    }
)

# Third Party
import numpy as np
import torch
from matplotlib import cm

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.camera import CameraObservation
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

simulation_app.update()
# Standard Library
import argparse

# Third Party
import carb
from helper import VoxelManager, add_robot_to_scene
from omni.isaac.core import World
from isaacsim.core.api.materials import OmniPBR
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.utils.types import ArticulationAction

# camera related imports
from omni.isaac.sensor import Camera
from pxr import Usd, UsdGeom, Gf

# CuRobo
from curobo.util.usd_helper import UsdHelper

parser = argparse.ArgumentParser()

parser.add_argument(
    "--show-window",
    action="store_true",
    help="When True, shows camera image in a CV window",
    default=False,
)

parser.add_argument("--robot", type=str, default="franka.yml", help="robot configuration to load")
parser.add_argument(
    "--use-debug-draw",
    action="store_true",
    help="When True, sets robot in static mode",
    default=False,
)
args = parser.parse_args()


def draw_points(voxels):
    # Third Party

    # Third Party
    try:
        from omni.isaac.debug_draw import _debug_draw
    except ImportError:
        from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
    # if draw.get_num_points() > 0:
    draw.clear_points()
    if len(voxels) == 0:
        return

    jet = cm.get_cmap("plasma").reversed()

    cpu_pos = voxels[..., :3].view(-1, 3).cpu().numpy()
    z_val = cpu_pos[:, 0]

    jet_colors = jet(z_val)

    b, _ = cpu_pos.shape
    point_list = []
    colors = []
    for i in range(b):
        # get list of points:
        point_list += [(cpu_pos[i, 0], cpu_pos[i, 1], cpu_pos[i, 2])]
        colors += [(jet_colors[i][0], jet_colors[i][1], jet_colors[i][2], 0.8)]
    sizes = [20.0 for _ in range(b)]

    draw.draw_points(point_list, colors, sizes)


def clip_camera(depth_tensor):
    """Clip camera image to bounding box for Isaac Sim camera"""
    h_ratio = 0.05
    w_ratio = 0.05

    if depth_tensor is None:
        return None

    h, w = depth_tensor.shape

    # Create a copy to avoid modifying original
    depth_clipped = depth_tensor.copy()

    # Clip borders
    depth_clipped[: int(h_ratio * h), :] = 0.0
    depth_clipped[int((1 - h_ratio) * h) :, :] = 0.0
    depth_clipped[:, : int(w_ratio * w)] = 0.0
    depth_clipped[:, int((1 - w_ratio) * w) :] = 0.0

    # Clip by distance
    depth_clipped[depth_clipped > clipping_distance] = 0.0

    return depth_clipped


def create_camera(camera_cfg, target_stage):
    cam_path = f"/World/panda/{camera_cfg['link']}/{camera_cfg['name']}"
    cam_prim = UsdGeom.Camera.Define(target_stage, cam_path)
    cam_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, clipping_distance))
    cam_prim.GetFocalLengthAttr().Set(_focallengthattr)
    cam_prim.GetFocusDistanceAttr().Set(_focusdistanceattr)

    # Set horizontal and vertical aperture for proper FOV
    cam_prim.GetHorizontalApertureAttr().Set(_horizontalapertureattr)
    cam_prim.GetVerticalApertureAttr().Set(_verticalapertureattr)

    # Set relative transform to link
    xform = UsdGeom.Xformable(cam_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(
        Gf.Vec3d(
            camera_cfg["translation"][0], camera_cfg["translation"][1], camera_cfg["translation"][2]
        )
    )
    xform.AddOrientOp().Set(
        Gf.Quatf(
            camera_cfg["orientation"][0],
            camera_cfg["orientation"][1],
            camera_cfg["orientation"][2],
            camera_cfg["orientation"][3],
        )
    )

    camera = Camera(
        prim_path=cam_path,
        name=camera_cfg["name"],
        frequency=camera_cfg["frequency"],
        resolution=_resolution,
    )
    return camera


if __name__ == "__main__":
    radius = 0.05
    act_distance = 0.4
    voxel_size = 0.05
    render_voxel_size = 0.02
    clipping_distance = 0.7
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    stage = my_world.stage
    my_world.scene.add_default_ground_plane()

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    target_material = OmniPBR("/World/looks/t", color=np.array([0, 1, 0]))
    target_material_2 = OmniPBR("/World/looks/t2", color=np.array([0, 1, 0]))

    target = cuboid.VisualCuboid(
        "/World/target_1",
        position=np.array([0.4, -0.5, 0.2]),
        orientation=np.array([0, 1.0, 0, 0]),
        size=0.04,
        visual_material=target_material,
    )

    # Make a target to follow
    target_2 = cuboid.VisualCuboid(
        "/World/target_2",
        position=np.array([0.4, 0.5, 0.2]),
        orientation=np.array([0.0, 1, 0.0, 0.0]),
        size=0.04,
        visual_material=target_material_2,
    )

    collision_checker_type = CollisionCheckerType.BLOX
    world_cfg = WorldConfig.from_dict(
        {
            "blox": {
                "world": {
                    "pose": [0, 0, 0, 1, 0, 0, 0],
                    "integrator_type": "occupancy",
                    "voxel_size": 0.02,
                }
            }
        }
    )
    tensor_args = TensorDeviceType()

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    robot, _ = add_robot_to_scene(robot_cfg, my_world, "/World/world_robot/")

    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_wall.yml"))
    )

    # 0 is table, 1 is wall
    world_cfg_table.cuboid[0].pose[2] -= 0.01
    world_cfg_table.cuboid[1].pose[0] += 1.15  # bring wall infront of the robot
    usd_help = UsdHelper()

    # common params for body cams
    _focallengthattr = 24.0
    _focusdistanceattr = 400.0
    _horizontalapertureattr = 20.955
    _verticalapertureattr = 15.2908
    _resolution = (640, 480)

    # TODO: fill values as noted in sketches
    cameras_config = {
        # front facing
        "cam1": {
            "name": "cam1",
            "link": "panda_link2",
            "translation": [0.0, 0.0, 0.0],
            "orientation": [1.0, 0.0, 0.0, 0.0],  # scalar first (w, x, y, z)
            "frequency": 30,
        },
        # left facing
        "cam2": {
            "name": "cam2",
            "link": "panda_link3",
            "translation": [0.0, 0.0, 0.0],
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "frequency": 30,
        },
        # right facing
        "cam3": {
            "name": "cam3",
            "link": "panda_link4",
            "translation": [0.0, 0.0, 0.0],
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "frequency": 30,
        },
        # left facing
        "cam4": {
            "name": "cam4",
            "link": "panda_link5",
            "translation": [0.0, 0.0, 0.0],
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "frequency": 30,
        },
        # front facing
        "cam5": {
            "name": "cam5",
            "link": "panda_link5",  # NOTE: 2 cameras on link5
            "translation": [0.0, 0.0, 0.0],
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "frequency": 30,
        },
        # right facing
        "cam6": {
            "name": "cam6",
            "link": "panda_link6",
            "translation": [0.0, 0.0, 0.0],
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "frequency": 30,
        },
        # front facing
        "cam7": {
            "name": "cam7",
            "link": "panda_link7",
            "translation": [0.0, 0.0, 0.0],
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "frequency": 30,
        },
    }

    # create body cameras
    body_cams = []
    for cam in cameras_config.values():
        body_cams.append(create_camera(cam, stage))

    usd_help.load_stage(my_world.stage)
    usd_help.add_world_to_stage(world_cfg_table.get_mesh_world(), base_frame="/World")
    world_cfg.add_obstacle(world_cfg_table.cuboid[0])
    # world_cfg.add_obstacle(world_cfg_table.cuboid[1])  # commented out to remove wall from collision check, since we want it to be detected from camera
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        trajopt_tsteps=32,
        collision_checker_type=CollisionCheckerType.BLOX,
        use_cuda_graph=True,
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        interpolation_dt=0.03,
        collision_activation_distance=0.025,
        acceleration_scale=1.0,
        self_collision_check=True,
        maximum_trajectory_dt=0.25,
        finetune_dt_scale=1.05,
        fixed_iters_trajopt=True,
        finetune_trajopt_iters=300,
        minimize_jerk=True,
    )
    motion_gen = MotionGen(motion_gen_config)
    print("warming up..")
    motion_gen.warmup(warmup_js_trajopt=False)

    world_model = motion_gen.world_collision

    i = 0
    target_list = [target, target_2]
    target_material_list = [target_material, target_material_2]
    for material in target_material_list:
        material.set_color(np.array([0.1, 0.1, 0.1]))
    target_idx = 0
    cmd_idx = 0
    cmd_plan = None
    articulation_controller = robot.get_articulation_controller()
    plan_config = MotionGenPlanConfig(
        enable_graph=False, enable_graph_attempt=4, max_attempts=2, enable_finetune_trajopt=True
    )
    if not args.use_debug_draw:
        voxel_viewer = VoxelManager(100, size=render_voxel_size)
    cmd_step_idx = 0
    body_cams_initialized = False

    while simulation_app.is_running():
        my_world.step(render=True)

        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            continue

        step_index = my_world.current_time_step_index

        if step_index <= 10:
            robot._articulation_view.initialize()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )

        # Initialize cameras after robot is initialized
        if step_index == 15 and not body_cams_initialized:
            # init body cams and depth
            for cam in body_cams:
                cam.initialize()
                cam.add_distance_to_image_plane_to_frame()
                cam.add_distance_to_camera_to_frame()
            body_cams_initialized = True
            print(f"{len(body_cams)} cameras initialized")

        if step_index % 5 == 0.0 and step_index > 20:  # Wait for cameras to be ready
            # Clear world model once before processing all cameras
            world_model.decay_layer("world")

            # Process data from ALL cameras
            valid_camera_count = 0
            all_camera_frames = []  # Store for visualization if needed

            for cam_idx, cam in enumerate(body_cams):
                frame_data = cam.get_current_frame()

                if frame_data is not None and "distance_to_image_plane" in frame_data:
                    valid_camera_count += 1

                    # Get depth image (using distance_to_camera for better accuracy)
                    depth_image = frame_data.get(
                        "distance_to_camera", frame_data.get("distance_to_image_plane")
                    )

                    # Clip and process depth
                    depth_clipped = clip_camera(depth_image)

                    if depth_clipped is not None:
                        # Convert to tensor
                        depth_tensor = (
                            torch.from_numpy(depth_clipped).float().to(tensor_args.device)
                        )

                        # Get camera pose from its world position
                        cam_position, cam_orientation = cam.get_world_pose()
                        camera_pose = Pose(
                            position=tensor_args.to_device(cam_position),
                            quaternion=tensor_args.to_device(cam_orientation),
                        )

                        # Get intrinsics
                        intrinsics = torch.tensor(cam.get_intrinsics_matrix()).to(
                            tensor_args.device
                        )

                        # Create camera observation
                        data_camera = CameraObservation(
                            depth_image=depth_tensor, intrinsics=intrinsics, pose=camera_pose
                        )

                        # Add this camera's frame to world model
                        world_model.add_camera_frame(data_camera, "world")

                        # Store frame for visualization
                        all_camera_frames.append(
                            {
                                "name": cam.name,
                                "rgb": frame_data.get("rgb"),
                                "depth": frame_data.get("distance_to_image_plane"),
                                "idx": cam_idx,
                            }
                        )

            # Process all camera frames together after adding them all
            if valid_camera_count > 0:
                world_model.process_camera_frames("world", False)
                torch.cuda.synchronize()
                world_model.update_blox_hashes()

                # Get voxels for visualization
                bounding = Cuboid("t", dims=[1, 1, 1.0], pose=[0, 0, 0, 1, 0, 0, 0])
                voxels = world_model.get_voxels_in_bounding_box(bounding, voxel_size)

                if voxels.shape[0] > 0:
                    voxels = voxels[voxels[:, 2] > voxel_size]
                    voxels = voxels[voxels[:, 0] > 0.0]
                    if args.use_debug_draw:
                        draw_points(voxels)
                    else:
                        voxels = voxels.cpu().numpy()
                        voxel_viewer.update_voxels(voxels[:, :3])
                else:
                    if not args.use_debug_draw:
                        voxel_viewer.clear()

                print(
                    f"Processed {valid_camera_count}/{len(body_cams)} cameras at step {step_index}"
                )

            # Display camera views if requested
            if args.show_window and len(all_camera_frames) > 0:
                # Option 1: Show grid of all cameras
                display_images = []

                for frame in all_camera_frames[:4]:  # Show max 4 cameras in grid
                    if frame["depth"] is not None and frame["rgb"] is not None:
                        # Convert depth to colormap
                        depth_colormap = cv2.applyColorMap(
                            cv2.convertScaleAbs(frame["depth"], alpha=100), cv2.COLORMAP_VIRIDIS
                        )

                        # RGB is in RGBA format, convert to RGB
                        rgb_display = frame["rgb"]
                        if rgb_display.shape[-1] == 4:
                            rgb_display = rgb_display[:, :, :3]

                        # Resize for grid display
                        rgb_small = cv2.resize(rgb_display, (320, 240))
                        depth_small = cv2.resize(depth_colormap, (320, 240))

                        # Add camera name label
                        cv2.putText(
                            rgb_small,
                            frame["name"],
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            depth_small,
                            f"{frame['name']}_depth",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )

                        # Stack horizontally for this camera
                        cam_pair = np.hstack((rgb_small, depth_small))
                        display_images.append(cam_pair)

                # Create grid layout
                if len(display_images) > 0:
                    # Stack vertically if multiple cameras
                    if len(display_images) == 1:
                        final_display = display_images[0]
                    elif len(display_images) == 2:
                        final_display = np.vstack(display_images)
                    else:
                        # 2x2 grid for 3-4 cameras
                        row1 = (
                            np.hstack(display_images[:2])
                            if len(display_images) >= 2
                            else display_images[0]
                        )
                        row2 = (
                            np.hstack(display_images[2:4])
                            if len(display_images) >= 3
                            else np.zeros_like(row1)
                        )
                        final_display = np.vstack((row1, row2))

                    cv2.namedWindow("Multi-Camera View", cv2.WINDOW_NORMAL)
                    cv2.imshow("Multi-Camera View", final_display)

                # Option 2: Cycle through cameras with keyboard
                key = cv2.waitKey(1)
                if key & 0xFF == ord("q") or key == 27:
                    cv2.destroyAllWindows()
                    break

        # Motion planning logic remains the same
        if cmd_plan is None and step_index % 10 == 0 and step_index > 20:
            # motion generation:
            for ks in range(len(target_material_list)):
                if ks == target_idx:
                    target_material_list[ks].set_color(np.ravel([0, 1.0, 0]))
                else:
                    target_material_list[ks].set_color(np.ravel([0.1, 0.1, 0.1]))

            sim_js = robot.get_joints_state()
            sim_js_names = robot.dof_names
            cu_js = JointState(
                position=tensor_args.to_device(sim_js.positions),
                velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
                acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
                jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
                joint_names=sim_js_names,
            )
            cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

            cube_position, cube_orientation = target_list[target_idx].get_world_pose()

            ik_goal = Pose(
                position=tensor_args.to_device(cube_position),
                quaternion=tensor_args.to_device(cube_orientation),
            )

            result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)

            succ = result.success.item()
            if succ:
                cmd_plan = result.get_interpolated_plan()
                cmd_plan = motion_gen.get_full_js(cmd_plan)

                idx_list = []
                common_js_names = []
                for x in sim_js_names:
                    if x in cmd_plan.joint_names:
                        idx_list.append(robot.get_dof_index(x))
                        common_js_names.append(x)

                cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)
                cmd_idx = 0
                target_idx += 1
                if target_idx >= len(target_list):
                    target_idx = 0
            else:
                carb.log_warn("Plan did not converge to a solution. No action is being taken.")

        if cmd_plan is not None:
            cmd_state = cmd_plan[cmd_idx]
            art_action = ArticulationAction(
                cmd_state.position.cpu().numpy(),
                cmd_state.velocity.cpu().numpy(),
                joint_indices=idx_list,
            )
            articulation_controller.apply_action(art_action)
            cmd_step_idx += 1
            if cmd_step_idx == 2:
                cmd_idx += 1
                cmd_step_idx = 0
            if cmd_idx >= len(cmd_plan.position):
                cmd_idx = 0
                cmd_plan = None

    print("finished program")
    simulation_app.close()
