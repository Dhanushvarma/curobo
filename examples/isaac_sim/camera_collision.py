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
import torch

a = torch.zeros(4, device="cuda:0")
# Third Party
import cv2
import numpy as np
import torch
from matplotlib import cm
from nvblox_torch.datasets.realsense_dataset import RealsenseDataloader
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "width": "1920",
        "height": "1080",
    }
)
# CuRobo
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

    jet = cm.get_cmap("plasma_r")

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
        # get list of points:
        point_list += [(cpu_pos[i, 0], cpu_pos[i, 1], cpu_pos[i, 2])]
        colors += [(jet_colors[i][0], jet_colors[i][1], jet_colors[i][2], 1.0)]
    sizes = [10.0 for _ in range(b)]

    draw.draw_points(point_list, colors, sizes)


def clip_camera(depth_tensor):
    if depth_tensor is None:
        return None
    h_ratio = 0.05
    w_ratio = 0.05
    h, w = depth_tensor.shape
    depth_clipped = depth_tensor.copy()
    depth_clipped[: int(h_ratio * h), :] = 0.0
    depth_clipped[int((1 - h_ratio) * h) :, :] = 0.0
    depth_clipped[:, : int(w_ratio * w)] = 0.0
    depth_clipped[:, int((1 - w_ratio) * w) :] = 0.0
    depth_clipped[depth_clipped > camera_optical_configuration["clipping_range"][1]] = 0.0

    return depth_tensor
    # return depth_clipped


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
    radius = 0.2
    act_distance = 0.4
    voxel_size = 0.025
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage
    my_world.scene.add_default_ground_plane()
    z_up = 1.0

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
        position=np.array([0.0, 0, 0.25 + z_up]),
        orientation=np.array([1, 0, 0, 0]),
        radius=radius,
        visual_material=target_material,
    )

    # constant for camera & camera_marker pose
    _t_vec = [0.0, -1.5, 0.25 + z_up]
    _r_vec = [0.7071068, 0.7071068, 0.0, 0.0]  # XYZ : 90 0 0

    # Camera marker for visualization (optional)
    camera_marker = cuboid.VisualCuboid(
        "/World/camera_marker",
        position=np.array(_t_vec),
        orientation=np.array(_r_vec),
        color=np.array([0.1, 0.1, 0.5]),
        size=0.03,
    )
    camera_marker.set_visibility(False)  # TODO: revent to false to hide marker

    # Create camera using USD API
    camera_path = "/World/main_camera"
    cam_prim = UsdGeom.Camera.Define(stage, camera_path)
    cam_prim.GetClippingRangeAttr().Set(Gf.Vec2f(*camera_optical_configuration["clipping_range"]))
    cam_prim.GetFocalLengthAttr().Set(camera_optical_configuration["focal_length"])
    cam_prim.GetFocusDistanceAttr().Set(camera_optical_configuration["focus_distance"])
    cam_prim.GetHorizontalApertureAttr().Set(camera_optical_configuration["horizontal_aperture"])
    cam_prim.GetVerticalApertureAttr().Set(camera_optical_configuration["vertical_aperture"])

    # Set camera position and orientation
    xform_cam = UsdGeom.Xformable(cam_prim)
    xform_cam.ClearXformOpOrder()
    xform_cam.AddTranslateOp().Set(Gf.Vec3d(_t_vec[0], _t_vec[1], _t_vec[2]))
    xform_cam.AddOrientOp().Set(
        Gf.Quatf(
            _r_vec[0],
            _r_vec[1],
            _r_vec[2],
            _r_vec[3],
        )
    )

    # Wrap with Isaac Sim Camera
    _camera = Camera(
        prim_path=camera_path,
        name="main_camera",
        frequency=30,
        resolution=camera_optical_configuration["resolution"],
    )

    # Load obstacles
    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "camera_wall.yml"))
    )
    world_cfg_table.cuboid[0].pose[2] += z_up

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
                    "integrator_type": "occupancy",
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
    tensor_args = TensorDeviceType()
    x_sph = torch.zeros((1, 1, 1, 4), device=tensor_args.device, dtype=tensor_args.dtype)
    x_sph[..., 3] = radius
    camera_initialized = False

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
                # Get depth image
                depth_image = frame_data["distance_to_image_plane"]
                depth_clipped = clip_camera(depth_image)

                if depth_clipped is not None:
                    # Convert to tensor
                    depth_tensor = torch.from_numpy(depth_clipped).float().to(tensor_args.device)

                    # Get camera intrinsics
                    intrinsics = torch.tensor(_camera.get_intrinsics_matrix()).to(
                        tensor_args.device
                    )

                    # Get camera pose, TODO: figure out which one is correct
                    cam_position, cam_orientation = _camera.get_world_pose()
                    # cam_position, cam_orientation = _camera.get_local_pose()

                    camera_pose = Pose(
                        position=tensor_args.to_device(cam_position),
                        quaternion=tensor_args.to_device(cam_orientation),
                    )

                    # Create camera observation
                    data_camera = CameraObservation(
                        depth_image=depth_tensor, intrinsics=intrinsics, pose=camera_pose
                    )

                    # Add to world model, TODO: make sure it has to be world frame!
                    model.world_model.add_camera_frame(data_camera, "world")
                    model.world_model.process_camera_frames("world", False)
                    torch.cuda.synchronize()
                    model.world_model.update_blox_hashes()

                    # Get voxels for visualization
                    bounding = Cuboid("t", dims=[1, 1, 1], pose=[0, 0, 0, 1, 0, 0, 0])
                    voxels = model.world_model.get_voxels_in_bounding_box(bounding, voxel_size)

                    if voxels is not None:
                        print("Number of voxels: ", voxels.shape[0])
                    draw_points(voxels)

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
                        print(f"Distance: {d.item():.4f}")
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
                        depth_colormap = cv2.applyColorMap(
                            cv2.convertScaleAbs(depth_display, alpha=100), cv2.COLORMAP_VIRIDIS
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
