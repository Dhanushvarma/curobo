#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Debugged version with fixes and diagnostic output
#


try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Third Party
import torch

a = torch.zeros(4, device="cuda:0")

import cv2
import numpy as np
import torch
from matplotlib import cm
import matplotlib.pyplot as plt
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "width": "1920",
        "height": "1080",
    }
)

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.camera import CameraObservation
from curobo.types.math import Pose
from curobo.util_file import get_world_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

simulation_app.update()
# Standard Library
import argparse

# Third Party
from omni.isaac.core import World
from isaacsim.core.api.materials import OmniPBR
from omni.isaac.core.objects import cuboid, sphere

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
parser.add_argument(
    "--debug",
    action="store_true",
    help="Enable debug output",
    default=True,
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

    # Use same colormap as realsense version
    jet = cm.get_cmap("plasma").reversed()

    cpu_pos = voxels[..., :3].view(-1, 3).cpu().numpy()
    z_val = cpu_pos[:, 1]
    # add smallest and largest values:
    # z_val = np.append(z_val, 1.0)
    # z_val = np.append(z_val,0.4)
    # scale values
    # z_val += 0.4
    # z_val[z_val>1.0] = 1.0
    # z_val = 1.0/z_val
    # z_val = z_val/1.5
    # z_val[z_val!=z_val] = 0.0
    # z_val[z_val==0.0] = 0.4

    jet_colors = jet(z_val)

    b, _ = cpu_pos.shape
    point_list = []
    colors = []
    for i in range(b):
        point_list += [(cpu_pos[i, 0], cpu_pos[i, 1], cpu_pos[i, 2])]
        colors += [(jet_colors[i][0], jet_colors[i][1], jet_colors[i][2], 1.0)]
    sizes = [10.0 for _ in range(b)]

    draw.draw_points(point_list, colors, sizes)


def clip_camera(depth_tensor, clipping_range):
    if depth_tensor is None:
        return None

    # Match realsense clipping ratios (15% instead of 5%)
    h_ratio = 0.15
    w_ratio = 0.15
    h, w = depth_tensor.shape
    depth_clipped = depth_tensor.copy()

    # Clip edges
    depth_clipped[: int(h_ratio * h), :] = 0.0
    depth_clipped[int((1 - h_ratio) * h) :, :] = 0.0
    depth_clipped[:, : int(w_ratio * w)] = 0.0
    depth_clipped[:, int((1 - w_ratio) * w) :] = 0.0

    # Clip by range
    depth_clipped[depth_clipped > clipping_range[1]] = 0.0
    depth_clipped[depth_clipped < clipping_range[0]] = 0.0

    return depth_clipped


def draw_line(start, gradient):
    # Third Party
    try:
        from omni.isaac.debug_draw import _debug_draw
    except ImportError:
        from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
    # if draw.get_num_points() > 0:
    draw.clear_lines()
    start_list = [start]
    end_list = [start + gradient]

    colors = [(0.0, 0, 0.8, 0.9)]

    sizes = [10.0]
    draw.draw_lines(start_list, end_list, colors, sizes)


if __name__ == "__main__":
    radius = 0.075
    act_distance = 0.4
    voxel_size = 0.03  # Match realsense voxel size
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage
    my_world.scene.add_default_ground_plane()
    z_up = 1.0
    tensor_args = TensorDeviceType()

    camera_optical_configuration = {
        "focal_length": 1.88,
        "focus_distance": 600.0,
        "horizontal_aperture": 2.58,
        "vertical_aperture": 1.60,
        "resolution": (640, 480),
        "clipping_range": (0.2, 2.0),
    }

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    target_material = OmniPBR("/World/looks/t", color=np.array([0, 1, 0]))

    target = sphere.VisualSphere(
        "/World/target",
        position=np.array([0.0, 0.25, 0.4]),  # Match realsense target position
        orientation=np.array([1, 0, 0, 0]),
        radius=radius,
        visual_material=target_material,
    )
    target.set_visibility(True)

    # Desired camera pose in world frame
    camera_desired_t_vec = torch.tensor([[0.0, -0.15, 0.4]])
    camera_desired_r_vec = torch.tensor([[0.7071068, 0.7071068, 0.0, 0.0]])
    camera_desired_pose = Pose(position=camera_desired_t_vec, quaternion=camera_desired_r_vec)

    # observed offset for IsaacSim API
    cam_offset = Pose(
        position=torch.tensor([[0.0, 0.0, 0.0]]),
        quaternion=torch.tensor([[0.5, 0.5, -0.5, -0.5]]),
    )  # quat -> (90 -90 0) XYZ Euler

    # (IsaacSim API sensor pose) * (cam_offset) = (camera_desired_pose)
    isaac_api_camera_pose: Pose = camera_desired_pose.multiply(cam_offset.inverse())

    # NVBlox pose = isaac_api_camera_pose * cam_nvblox_offset
    cam_nvblox_offset = Pose(
        position=tensor_args.to_device(torch.tensor([[0.0, 0.0, 0.0]])),
        quaternion=tensor_args.to_device(torch.tensor([[+0.5, -0.5, +0.5, -0.5]])),
    )  # quat -> (-90 90 0) XYZ Euler

    # make camera from Isaac Sim API
    _camera = Camera(
        prim_path="/World/main_camera",
        name="main_camera",
        frequency=30,
        position=isaac_api_camera_pose.position.tolist()[0],
        orientation=isaac_api_camera_pose.quaternion.tolist()[0],
        resolution=camera_optical_configuration["resolution"],
    )
    _camera.initialize()  # need to initialize before setting parameters
    _camera.set_focal_length(camera_optical_configuration["focal_length"])
    _camera.set_focus_distance(camera_optical_configuration["focus_distance"])
    _camera.set_horizontal_aperture(camera_optical_configuration["horizontal_aperture"])
    _camera.set_vertical_aperture(camera_optical_configuration["vertical_aperture"])
    _camera.set_clipping_range(
        camera_optical_configuration["clipping_range"][0],
        camera_optical_configuration["clipping_range"][1],
    )

    if False:
        # Camera marker for visualization (optional)
        camera_marker = cuboid.VisualCuboid(
            "/World/camera_marker",
            position=np.array(_t_vec),
            orientation=np.array(_r_vec_marker),
            color=np.array([0.1, 0.1, 0.5]),
            size=0.03,
        )
        camera_marker.set_visibility(False)  # Make visible for debugging

    # Load obstacles
    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "camera_wall.yml"))
    )

    # Add obstacles to scene
    usd_help = UsdHelper()
    usd_help.load_stage(my_world.stage)
    usd_help.add_world_to_stage(world_cfg_table.get_mesh_world(), base_frame="/World")

    # Setup collision world
    collision_checker_type = CollisionCheckerType.BLOX
    world_cfg = WorldConfig.from_dict(
        {
            "blox": {
                "world": {
                    "pose": [0, 0, 0, 1, 0, 0, 0],
                    "integrator_type": "tsdf",
                    "voxel_size": 0.03,
                }
            }
        }
    )

    # Add obstacles to collision world
    world_cfg.add_obstacle(world_cfg_table.cuboid[0])

    config = RobotWorldConfig.load_from_config(
        "franka.yml",
        world_cfg,
        collision_activation_distance=act_distance,
        collision_checker_type=collision_checker_type,
    )

    model = RobotWorld(config)

    i = 0
    x_sph = torch.zeros((1, 1, 1, 4), device=tensor_args.device, dtype=tensor_args.dtype)
    x_sph[..., 3] = radius
    camera_initialized = False
    frame_count = 0

    while simulation_app.is_running():
        my_world.step(render=True)

        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            continue

        step_index = my_world.current_time_step_index

        # Initialize camera once after simulation starts
        if step_index == 10 and not camera_initialized:
            _camera.initialize()
            _camera.add_distance_to_image_plane_to_frame()
            _camera.add_distance_to_camera_to_frame()
            camera_initialized = True
            print("Camera initialized")
            continue

        # Process camera data after initialization
        if camera_initialized and step_index > 15:
            # Get target position
            sph_position, _ = target.get_local_pose()
            x_sph[..., :3] = tensor_args.to_device(sph_position).view(1, 1, 1, 3)

            # Decay world model
            model.world_model.decay_layer("world")

            # Get camera data
            frame_data = _camera.get_current_frame()

            if frame_data is not None and "distance_to_image_plane" in frame_data:
                frame_count += 1

                # Get depth image
                depth_image = frame_data["distance_to_image_plane"]
                depth_image[np.isinf(depth_image)] = 0  # making inf values zero
                depth_image[np.isnan(depth_image)] = 0  # handle NaN values

                # Debug: Check depth statistics before clipping
                if args.debug and frame_count % 30 == 0:
                    valid_depth = depth_image[depth_image > 0]
                    if len(valid_depth) > 0:
                        print(f"\n--- Frame {frame_count} Debug Info ---")
                        print(f"Depth range: {valid_depth.min():.3f} - {valid_depth.max():.3f}")
                        print(f"Valid depth pixels: {len(valid_depth)} / {depth_image.size}")

                depth_clipped = clip_camera(
                    depth_image, camera_optical_configuration["clipping_range"]
                )

                if depth_clipped is not None:
                    # Convert to tensor
                    depth_tensor = torch.from_numpy(depth_clipped).float().to(tensor_args.device)

                    # Debug: Check depth tensor
                    if args.debug and frame_count % 30 == 0:
                        valid_tensor = depth_tensor[depth_tensor > 0]
                        print(f"Depth tensor valid pixels: {valid_tensor.numel()}")
                        if valid_tensor.numel() > 0:
                            print(
                                f"Depth tensor range: {valid_tensor.min():.3f} - {valid_tensor.max():.3f}"
                            )

                    # Get camera intrinsics
                    intrinsics = torch.tensor(_camera.get_intrinsics_matrix()).to(
                        tensor_args.device
                    )

                    # IsaacSim Sensors API camera pose
                    cam_position, cam_orientation = _camera.get_local_pose()

                    # convert to cuRobo Pose
                    camera_pose = Pose(
                        position=tensor_args.to_device(torch.from_numpy(cam_position).unsqueeze(0)),
                        quaternion=tensor_args.to_device(
                            torch.from_numpy(cam_orientation).unsqueeze(0)
                        ),
                    )

                    # camera pose for NVBlox = camera_pose * cam_nvblox_offset
                    camera_pose_nvblox: Pose = camera_pose.multiply(cam_nvblox_offset)

                    # Debug: Print camera pose
                    if args.debug and frame_count % 30 == 0:
                        print(50 * "-")
                        print(f"Camera position: {cam_position}")
                        print(f"Camera orientation: {cam_orientation}")
                        print(f"NvBlox Camera position: {camera_pose_nvblox.position}")
                        print(f"NvBlox Camera orientation: {camera_pose_nvblox.quaternion}")

                    # Create camera observation
                    data_camera = CameraObservation(
                        depth_image=depth_tensor, intrinsics=intrinsics, pose=camera_pose_nvblox
                    )
                    data_camera.to(device=model.tensor_args.device)

                    # Add to world model
                    model.world_model.add_camera_frame(data_camera, "world")
                    model.world_model.process_camera_frames("world", False)
                    torch.cuda.synchronize()
                    model.world_model.update_blox_hashes()

                    # Get voxels for visualization
                    bounding = Cuboid("t", dims=[1, 1, 1], pose=[0, 0, 0, 1, 0, 0, 0])
                    voxels = model.world_model.get_voxels_in_bounding_box(bounding, voxel_size)

                    if voxels is not None:
                        num_voxels = voxels.shape[0] if len(voxels.shape) > 0 else 0
                        if args.debug and frame_count % 30 == 0:
                            print(f"Number of voxels: {num_voxels}")
                            if num_voxels > 0:
                                voxel_pos = voxels[..., :3].view(-1, 3)
                                print(f"Voxel position range:")
                                print(
                                    f"  X: {voxel_pos[:, 0].min():.3f} - {voxel_pos[:, 0].max():.3f}"
                                )
                                print(
                                    f"  Y: {voxel_pos[:, 1].min():.3f} - {voxel_pos[:, 1].max():.3f}"
                                )
                                print(
                                    f"  Z: {voxel_pos[:, 2].min():.3f} - {voxel_pos[:, 2].max():.3f}"
                                )
                        draw_points(voxels)
                    else:
                        if args.debug and frame_count % 30 == 0:
                            print("No voxels generated!")

                    # Check collision
                    d, d_vec = model.get_collision_vector(x_sph)

                    # Update target color based on collision distance
                    p = max(1, d.item() * 5)
                    if d.item() == 0.0:
                        target_material.set_color(np.ravel([0, 1, 0]))
                    elif d.item() <= model.contact_distance:
                        target_material.set_color(np.array([0, 0, p]))
                    else:
                        target_material.set_color(np.array([p, 0, 0]))

                    if d.item() != 0.0:
                        if args.debug and frame_count % 30 == 0:
                            print(f"Collision distance: {d.item():.4f}")
                        draw_line(sph_position, d_vec[..., :3].view(3).cpu().numpy())
                    else:
                        # Clear lines when no collision
                        try:
                            from omni.isaac.debug_draw import _debug_draw
                        except ImportError:
                            from isaacsim.util.debug_draw import _debug_draw
                        draw = _debug_draw.acquire_debug_draw_interface()
                        draw.clear_lines()

                # Display camera view if requested
                if args.show_window and frame_data is not None:
                    depth_display = frame_data.get("distance_to_image_plane")
                    rgb_display = frame_data.get("rgb")

                    if depth_display is not None:
                        # Normalize depth for visualization
                        depth_normalized = depth_display.copy()
                        depth_normalized[
                            depth_normalized > camera_optical_configuration["clipping_range"][1]
                        ] = 0
                        depth_colormap = cv2.applyColorMap(
                            cv2.convertScaleAbs(depth_normalized, alpha=255), cv2.COLORMAP_VIRIDIS
                        )

                        # Show images
                        if rgb_display is not None:
                            if rgb_display.shape[-1] == 4:
                                rgb_display = rgb_display[:, :, :3]
                            images = np.hstack((rgb_display, depth_colormap))
                        else:
                            images = depth_colormap

                        cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)
                        cv2.imshow("Camera View", images)
                        key = cv2.waitKey(1)

                        if key & 0xFF == ord("q") or key == 27:
                            cv2.destroyAllWindows()
                            break

    print("finished program")
    simulation_app.close()
