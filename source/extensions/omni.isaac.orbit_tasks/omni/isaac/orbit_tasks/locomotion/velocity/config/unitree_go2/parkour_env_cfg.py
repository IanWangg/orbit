"""
Cfgs for ParkourEnv
"""
from __future__ import annotations

import omni.isaac.orbit.sim as sim_utils

from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.terrains.height_field.utils import height_field_to_mesh

import math
from dataclasses import MISSING

from omni.isaac.orbit_tasks.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    MySceneCfg,
    CommandsCfg,
    ObservationsCfg,
    RewardsCfg
)
from omni.isaac.orbit_assets.unitree import UNITREE_GO2_CFG  # isort: skip
import omni.isaac.orbit.terrains as terrain_gen
from omni.isaac.orbit.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from omni.isaac.orbit.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from omni.isaac.orbit.sensors import ContactSensorCfg, RayCasterCfg, patterns

from omni.isaac.orbit.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg,
)
from omni.isaac.orbit_tasks.locomotion.velocity import mdp
from omni.isaac.orbit.utils.noise import AdditiveUniformNoiseCfg as Unoise

import numpy as np

TERRAIN_LENGTH = 12.0
WALL_HEIGHT = 1.0

@height_field_to_mesh
def cross_obstacle_terrain(difficulty: float, cfg: HfCrossObstacleTerrainCfg) -> np.ndarray:
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    height_min = int(cfg.height_min / cfg.vertical_scale)
    height_max = int(cfg.height_max / cfg.vertical_scale)

    obstacle_pixels_start = int(cfg.obstacle_start / cfg.horizontal_scale)
    obstacle_pixels_end = int((cfg.obstacle_start + cfg.obstacle_width) / cfg.horizontal_scale)

    # set the default hf surface
    hf_raw = np.zeros((width_pixels, length_pixels), dtype=np.float32)

    # add the obstacle, which should occupy the entire width of the terrain
    height = height_min + (height_max - height_min) * difficulty
    hf_raw[:, obstacle_pixels_start:obstacle_pixels_end] = height

    # add the walls around the obstacle
    if cfg.around_wall:
        wall_height = int(cfg.wall_height / cfg.vertical_scale)
        wall_width = int(cfg.wall_width / cfg.horizontal_scale)
        hf_raw[:wall_width, :] = wall_height
        hf_raw[-wall_width:, :] = wall_height
        hf_raw[:, :wall_width] = wall_height
        hf_raw[:, -wall_width:] = wall_height

    return hf_raw

@height_field_to_mesh
def leap_through_gap_terrain(difficulty: float, cfg: HfLeapThroughGapTerrainCfg) -> np.ndarray:
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    height_min = int(cfg.height_min / cfg.vertical_scale)
    height_max = int(cfg.height_max / cfg.vertical_scale)

    # gap width
    gap_pixels_start = int(cfg.obstacle_start / cfg.horizontal_scale)
    gap_width_pixels = cfg.obstacle_width + (cfg.obstacle_width_max - cfg.obstacle_width) * difficulty
    gap_pixels_end = int((cfg.obstacle_start + gap_width_pixels) / cfg.horizontal_scale)

    # set the default hf surface
    hf_raw = np.zeros((width_pixels, length_pixels), dtype=np.float32)

    # add the obstacle, which should occupy the entire width of the terrain
    height = np.random.uniform(height_min, height_max)
    hf_raw[:, gap_pixels_start:gap_pixels_end] = height

    # add the walls around the obstacle
    if cfg.around_wall:
        wall_height = int(cfg.wall_height / cfg.vertical_scale)
        wall_width = int(cfg.wall_width / cfg.horizontal_scale)
        hf_raw[:wall_width, :] = wall_height
        hf_raw[-wall_width:, :] = wall_height
        hf_raw[:, :wall_width] = wall_height
        hf_raw[:, -wall_width:] = wall_height

    return hf_raw

@configclass
class HfCrossObstacleTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a pyramid stairs height field terrain."""
    
    function = cross_obstacle_terrain
    
    around_wall: bool = True
    wall_width: float = 0.1
    wall_height: float = WALL_HEIGHT

    base_height: float = 0.0

    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float | None = None

    height_max: float = MISSING
    height_min: float = MISSING
    obstacle_width: float = MISSING # in m

    # the obstacle is a 1D array of floats, where each float is the height of the obstacle at that position
    # right now only put one obstacle in the middle of the terrain
    obstacle_start: float = MISSING # in m

@configclass
class HfLeapThroughGapTerrainCfg(HfCrossObstacleTerrainCfg):
    """Configuration for a pyramid stairs height field terrain."""
    
    function = leap_through_gap_terrain
    
    obstacle_width_max: float = MISSING # in m

@configclass
class HfSlopeThenJumpTerrainCfg(HfTerrainBaseCfg):
    function = cross_obstacle_terrain
    
    around_wall: bool = True
    wall_width: float = 0.1
    wall_height: float = WALL_HEIGHT

    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float | None = None

    height_max: float = MISSING
    height_min: float = MISSING

    slope_length: float = MISSING

# Define new terrain config
PARKOUR_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(6.0, TERRAIN_LENGTH), # (x, y), size of each sub-terrain,
    border_width=20.0, # border width of the whole terrain, this is not important, we will add strict boarder on each sub-terrain
    num_rows=5, # number of rows of sub-terrains
    num_cols=5, # number of columns of sub-terrains
    color_scheme="height",
    horizontal_scale=0.1, # original setup
    vertical_scale=0.005, # original setup
    slope_threshold=None, # No vertical adjustment
    use_cache=False,
    sub_terrains={
        "leap": HfLeapThroughGapTerrainCfg(
            proportion=0.3,
            height_min=-3.0,
            height_max=-5.0,
            obstacle_width=0.1,
            obstacle_width_max=0.5,
            obstacle_start=TERRAIN_LENGTH // 2
        ),
        "narrow_climb": HfCrossObstacleTerrainCfg(
            proportion=0.3,
            height_min=0.1,
            height_max=0.4,
            obstacle_width=0.5,
            obstacle_start=TERRAIN_LENGTH // 2
        ),
        "wide_climb": HfCrossObstacleTerrainCfg(
            proportion=0.3,
            height_min=0.1,
            height_max=0.4,
            obstacle_width=TERRAIN_LENGTH // 4,
            obstacle_start=TERRAIN_LENGTH // 2
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.01, 0.03), noise_step=0.005, border_width=0.1
        ),
    }
)


# Override the default scene config
@configclass
class ParkourSceneCfg(MySceneCfg):
    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=PARKOUR_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    """
    Lidar sensor 
    """
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.BpearlPatternCfg(
            horizontal_fov=360.0, 
            vertical_ray_angles=[0.0]
        ),
        debug_vis=True,
        mesh_prim_paths=["/World/ground/terrain/mesh"],
    )

    """
    Forward facing sensor
    """
    forward_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(2.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.5, size=[2.0, 2.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

@configclass
class ParkourObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-2.0, 2.0),
        )
        
        lidar_scan = ObsTerm(
            func=mdp.lidar_scan,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(0.0, 3.0),
        )

        forward_height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("forward_height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-2.0, 2.0),
        )


        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


"""
TODO: Add the reward configuration:
1. jump reward:
    - When leap through the gap, give a reward for the height of the robot
    i. This reward should be only given when robot is above the gap, this requires the detection of the relative height of the robot
        a. This can be done by give reward to the robot when the height scanner (which measures the relative height of the robot)
            detects the robot is "way above" the ground, this would mean the robot is above the gap
    ii. The reward should be proportional to the height of the robot
"""

"""
TODO: Add the termination configuration:
1. When the robot falls into the gap, terminate the episode:
    - This can be done by checking the absolute height of the robot
2. When the robot hit the wall, terminate the episode
3. When the robot reaches certain area, terminate the episode
"""
        

@configclass
class UnitreeGo2ParkourEnvCfg(LocomotionVelocityRoughEnvCfg):
    
    # update scene setting
    scene = ParkourSceneCfg(num_envs=64, env_spacing=2.5)
    observations = ParkourObservationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # randomization
        self.randomization.push_robot = None
        self.randomization.add_base_mass.params["mass_range"] = (-1.0, 3.0)
        self.randomization.add_base_mass.params["asset_cfg"].body_names = "base"
        self.randomization.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.randomization.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.randomization.reset_base.params = {
            # "pose_range": {"x": (-0.5, 0.5), "y": (8.0, 9.0), "yaw": (-3.14, 3.14)},
            "pose_range": {"x": (-1.0, 1.0), "y": (TERRAIN_LENGTH // 2 - 2.0, TERRAIN_LENGTH // 2 - 1.0), "yaw": (-1, -1)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # update commands
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0) # forward vel
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0) # side vel
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

@configclass
class UnitreeGo2ParkourEnvCfg_PLAY(UnitreeGo2ParkourEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.randomization.base_external_force_torque = None
        self.randomization.push_robot = None