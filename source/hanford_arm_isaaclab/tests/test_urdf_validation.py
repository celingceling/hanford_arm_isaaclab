
"""
URDF Validation Test Suite for Hanford Arm
Tests: joint limits, axes, collision, EE frame, deterministic control

Usage:
    %ISAACLAB_EXE% -p source/hanford_arm_isaaclab/tests/test_urdf_validation.py --headless
"""

# ============================================================================
# IMPORTS
# ============================================================================
import argparse
from multiprocessing import get_context
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path

# Isaac Sim imports
from isaaclab.app import AppLauncher

# ============================================================================
# Define asset paths
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
USD_PATH = str(PROJECT_ROOT / "assets" / "hanford_arm_moveit_isaaclab_another.usd")


# Prim paths
robot_prim_path = "/World/pit_robot_assembly"
ptz_prim_path = "/World/scope89_ptz/base_link"

# Parse args before importing Isaac modules
parser = argparse.ArgumentParser(description="URDF Validation Tests")
parser.add_argument("--headless", action="store_true", help="Run headless")
parser.add_argument("--usd_path", type=str, default=USD_PATH, help="Path to robot USD file")
parser.add_argument("--duration", type=float, default=60.0, help="Stability test duration (sec)")
parser.add_argument("--output_dir", type=str, default="logs/urdf_validation", help="Output directory")
args_cli = parser.parse_args()

# ============================================================================
# LAUNCH ISAAC SIM
# ============================================================================
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Post-launch imports
import isaaclab.sim.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaacsim.core.api.world import World
from isaaclab.sim import SimulationContext
from isaaclab.sim.simulation_context import set_camera_view
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.seed import configure_seed
# from isaaclab.utils.math import quat_from_euler_xyz, matrix_from_quat

# from pxr import Usd
# stage = Usd.Stage.Open(r"C:\Users\LICF\projects\hanford_arm_isaaclab\assets\hanford_arm_moveit_isaaclab_another.usd")
# print("Default prim:", stage.GetDefaultPrim())
# print("Up axis:", stage.GetMetadata("upAxis"))

# # Check Isaac Sim schema version metadata
# root = stage.GetPseudoRoot()
# for prim in stage.Traverse():
#     print(prim.GetPath(), prim.GetTypeName(), prim.GetSpecifier())

# ============================================================================
# CONFIGURATION CLASS
# ============================================================================


class URDFValidationConfig:
    """Test configuration - easily extensible to multi-env"""

    # Simulation
    num_envs: int = 1  # Single env for now, but structure supports scaling
    sim_dt: float = 0.01  # 100Hz physics
    decimation: int = 2   # 50Hz control

    # Asset paths
    usd_path: str = USD_PATH
    output_dir: str = "logs/urdf_validation"

    # Test parameters
    test_duration: float = 600.0  # 10 min
    position_tolerance: float = 0.01  # rad
    velocity_tolerance: float = 0.05  # rad/s
    contact_force_threshold: float = 5.0 # N

    # Joint control (tune for robot)
    stiffness: float = 400.0
    damping: float = 40.0

    # Collision filtering
    collision_group: int = 0  # Example: filter self-collisions


# ============================================================================
# TEST FIXTURE CLASS
# ============================================================================


class URDFValidator:
    """URDF validation test suite"""

    def __init__(self, cfg: URDFValidationConfig):
        self.cfg = cfg
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Create output directory
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        # Will be initialized in setup()
        self.sim = None
        self.sim_cfg = None
        self.robot = None
        self.contact_sensor = None
        self.num_joints = None
        self.joint_names = None
        self.joint_limits = None
        
        self.robot_prim_path = robot_prim_path
        self.ptz_prim_path = ptz_prim_path

    # ------------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------------
    def setup(self):
        """Initialize simulation and robot"""
        print("\n" + "="*60)
        print("SETUP: Initializing simulation")
        print("="*60)

        # Create simulation context
        self.sim_cfg = sim_utils.SimulationCfg(
            dt=self.cfg.sim_dt,
            device=self.device,
        )
        
        self.sim = SimulationContext(self.sim_cfg)
        
        # Setup scene
        set_camera_view(
            eye=[2.5, 2.5, 2.5], 
            target=[0.0, 0.0, 0.5] # TODO start at a better view
            ) 

        # Create ground plane
        cfg_ground = sim_utils.GroundPlaneCfg()
        cfg_ground.func("/World/Ground", cfg_ground)
        
        # # # Add existing usd to this new stage we are making
        # from isaacsim.core.utils.stage import add_reference_to_stage
        # add_reference_to_stage(
        #     usd_path=USD_PATH, 
        #     prim_path="/World/Robot"
        #     )
        
        ## DEBUG
        # After creating robot_cfg, before Articulation()
        
        from pxr import Usd
        stage = simulation_app.context.get_stage()

        # Check what's in the USD
        # usd_stage = Usd.Stage.Open(self.cfg.usd_path)
        # print(f"Default prim: {usd_stage.GetDefaultPrim()}")
        # print(f"Root prims: {[p.GetPath() for p in usd_stage.GetPseudoRoot().GetChildren()]}")
        # usd_stage = None

        # # After sim.reset() fails, check current stage
        # print(f"Current stage prims: {[p.GetPath() for p in stage.Traverse()]}")

        # # Check if articulation exists at expected path
        # prim = stage.GetPrimAtPath("/World/Robot")
        # print(f"Prim valid: {prim.IsValid()}")
        # if prim.IsValid():
        #     print(f"Prim type: {prim.GetTypeName()}")
            
        # for p in stage.Traverse():
        #     if "pit_robot" in str(p.GetPath()) or "base_link" in str(p.GetPath()):
        #         print(f"Found: {p.GetPath()}")
            
        ## END DEBUG
        self.sim.reset()
        
        # Create articulation
        # robot_cfg = ArticulationCfg(
        #     prim_path="/World/Robot/pit_robot_robotonly/base_link",  # This should match the path INSIDE the USD
        #     spawn=None,
        #     init_state=ArticulationCfg.InitialStateCfg(
        #         pos=(0.0, 0.0, 0.0),
        #         rot=(1.0, 0.0, 0.0, 0.0),
        #     ),
        #     actuators={
        #         "joints": ImplicitActuatorCfg(
        #             joint_names_expr=[".*"],
        #             stiffness=self.cfg.stiffness,
        #             damping=self.cfg.damping,
        #         ),
        #     },
        # )
        
        robot_cfg = ArticulationCfg(
            prim_path="/World/Robot/pit_robot_robotonly/pipe_entry",
            spawn=sim_utils.UsdFileCfg(
                usd_path=USD_PATH,
                activate_contact_sensors=True,
            ),
            actuators={
                "joints": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=self.cfg.stiffness,
                    damping=self.cfg.damping,
                ),
            },
        )
        self.robot = Articulation(robot_cfg)
        
        stage = simulation_app.context.get_stage()
        print("\n=== RIGID BODIES ===")
        for p in stage.Traverse():
            if "Robot" in str(p.GetPath()):
                schemas = p.GetAppliedSchemas()
                if any("RigidBody" in s for s in schemas):
                    print(f"  RigidBody: {p.GetPath()} [{p.GetTypeName()}]")
                elif any("ArticulationRoot" in s for s in schemas):
                    print(f"  ArticulationRoot: {p.GetPath()} [{p.GetTypeName()}]")
        print("=== END ===")

        # Setup contact sensor for collision detection
        contact_cfg = ContactSensorCfg(
            prim_path="/World/Robot/pit_robot_robotonly/pipe_entry/.*", # put sensors on all links
            update_period=0.0,
            history_length=1,
            filter_prim_paths_expr=["/World/Ground"],
        )
        self.contact_sensor = ContactSensor(contact_cfg)

        # Reset and play
        self.sim.reset()
        
        print("Contact sensor prims:", self.contact_sensor.body_names)
        print("Net forces tensor:", self.contact_sensor.data.net_forces_w)

        # Get joint information
        self.num_joints = self.robot.num_joints
        self.joint_names = self.robot.joint_names
        self.joint_limits = torch.stack([
            self.robot.data.soft_joint_pos_limits[0, :, 0],  # lower
            self.robot.data.soft_joint_pos_limits[0, :, 1],  # upper
        ], dim=0)

        print(f"✓ Robot loaded: {self.num_joints} joints")
        print(f"  Joint names: {self.joint_names}")
        print(f"  Device: {self.device}")

    # ------------------------------------------------------------------------
    # VALIDATION TESTS
    # ------------------------------------------------------------------------
    def test_joint_limits(self) -> dict:
        """Test 1: Validate joint limits"""
        print("\n" + "="*60)
        print("TEST 1: Joint Limits Validation")
        print("="*60)

        results = {}

        for i, name in enumerate(self.joint_names):
            lower = self.joint_limits[0, i].item()
            upper = self.joint_limits[1, i].item()

            # Test commanding beyond limits
            test_positions = torch.zeros(1, self.num_joints, device=self.device)

            # Test lower limit
            test_positions[0, i] = lower - 0.5
            self.robot.set_joint_position_target(target=test_positions)
            for _ in range(10):
                self.sim.step()
            actual_lower = self.robot.data.joint_pos[0, i].item()

            # Test upper limit
            test_positions[0, i] = upper + 0.5
            self.robot.set_joint_position_target(target=test_positions)
            for _ in range(10):
                self.sim.step()
            actual_upper = self.robot.data.joint_pos[0, i].item()

            # Reset
            test_positions[0, i] = 0.0
            self.robot.set_joint_position_target(target=test_positions)
            for _ in range(10):
                self.sim.step()

            clamped_correctly = (actual_lower >= lower - 0.1) and (actual_upper <= upper + 0.1)

            results[name] = {
                'lower_limit': lower,
                'upper_limit': upper,
                'clamped_correctly': clamped_correctly
            }

            status = "✓" if clamped_correctly else "✗"
            print(f"  {status} {name:20s}: [{lower:7.3f}, {upper:7.3f}] rad")

        return results

    def test_joint_axes(self) -> dict:
        """Test 2: Validate joint axes by measuring motion"""
        print("\n" + "="*60)
        print("TEST 2: Joint Axes Validation (Motion Test)")
        print("="*60)

        results = {}

        # Reset to zero configuration
        zero_pos = torch.zeros(1, self.num_joints, device=self.device)
        self.robot.set_joint_position_target(zero_pos)
        for _ in range(50):
            self.sim.step()
        ee_idx = self.robot.body_names.index("end_effector")
        initial_ee_pos = self.robot.data.body_pos_w[0, ee_idx].clone()  # Assuming last body is EE

        for i, name in enumerate(self.joint_names):
            # Apply small positive delta
            test_pos = zero_pos.clone()
            delta = 0.1  # rad
            test_pos[0, i] = delta

            self.robot.set_joint_position_target(test_pos)
            for _ in range(50):
                self.sim.step()

            # Measure EE displacement
            final_ee_pos = self.robot.data.body_pos_w[0, ee_idx]
            displacement = (final_ee_pos - initial_ee_pos).cpu().numpy()

            # Reset
            self.robot.set_joint_position_target(zero_pos)
            for _ in range(50):
                self.sim.step()

            results[name] = {
                'delta_commanded': delta,
                'ee_displacement': displacement,
                'displacement_magnitude': np.linalg.norm(displacement)
            }

            print(f"  {name:20s}: ΔEE = [{displacement[0]:7.4f}, {displacement[1]:7.4f}, {displacement[2]:7.4f}] m")

        return results

    def test_ee_frame(self) -> dict:
        """Test 3: Validate end-effector frame exists and is accessible"""
        print("\n" + "="*60)
        print("TEST 3: End-Effector Frame Validation")
        print("="*60)

        # Reset to zero
        zero_pos = torch.zeros(1, self.num_joints, device=self.device)
        self.robot.set_joint_position_target(target=zero_pos)
        for _ in range(50):
            self.sim.step()

        # Get EE pose
        ee_idx = self.robot.body_names.index("end_effector")
        ee_pos = self.robot.data.body_pos_w[0, ee_idx].cpu().numpy()
        ee_quat = self.robot.data.body_quat_w[0, ee_idx].cpu().numpy()

        results = {
            'ee_position': ee_pos,
            'ee_quaternion': ee_quat,
            'accessible': True
        }

        print(f"  ✓ EE Position: [{ee_pos[0]:7.4f}, {ee_pos[1]:7.4f}, {ee_pos[2]:7.4f}] m")
        print(f"  ✓ EE Quaternion: [{ee_quat[0]:7.4f}, {ee_quat[1]:7.4f}, {ee_quat[2]:7.4f}, {ee_quat[3]:7.4f}]")
        print("  Note: FK comparison requires URDF file - skipped")

        return results

    def test_hold_pose_stability(self) -> dict:
        """Test 4: Hold pose with PD control"""
        print("\n" + "="*60)
        print(f"TEST 4: Pose Stability ({self.cfg.test_duration}s)")
        print("="*60)

        # Target: mid-range positions
        target_pos = torch.zeros(1, self.num_joints, device=self.device)
        for i in range(self.num_joints):
            mid = (self.joint_limits[0, i] + self.joint_limits[1, i]) / 2.0
            target_pos[0, i] = mid * 0.3 + self.joint_limits[0, i]

        print(f"  Target positions: {target_pos[0].cpu().numpy()}")
        print(f"  Running for {self.cfg.test_duration}s...")

        metrics = self.run_control_loop(target_pos, self.cfg.test_duration)

        # Analyze stability
        pos_errors = np.array(metrics['pos_error'])
        max_error = np.max(np.abs(pos_errors))
        final_errors = pos_errors[-10:]  # Last 10 samples
        steady_state_error = np.mean(np.abs(final_errors))

        stable = steady_state_error < self.cfg.position_tolerance

        results = {
            'max_error': max_error,
            'steady_state_error': steady_state_error,
            'stable': stable,
            'metrics': metrics
        }

        status = "✓" if stable else "✗"
        print(f"  {status} Max error: {max_error:.5f} rad")
        print(f"  {status} Steady-state error: {steady_state_error:.5f} rad")
        print(f"  {status} Stable: {stable} (threshold: {self.cfg.position_tolerance})")

        assert stable, f"Stability test failed: {steady_state_error:.5f} > {self.cfg.position_tolerance}"

        return results

    def test_collision_detection(self) -> dict:
        """Test 5: Collision detection"""
        print("\n" + "="*60)
        print("TEST 5: Collision Detection")
        print("="*60)

        # Move to extreme position (may cause self-collision)
        extreme_pos = torch.zeros(1, self.num_joints, device=self.device)
        for i in range(self.num_joints):
            extreme_pos[0, i] = self.joint_limits[1, i] * 0.9  # 90% of upper limit

        self.robot.set_joint_position_target(extreme_pos)

        contact_detected = False
        max_force = 0.0

        for _ in range(100):
            self.sim.step()
            self.contact_sensor.update(self.sim_cfg.dt)

            forces = self.contact_sensor.data.net_forces_w
            if forces is not None and len(forces) > 0:
                max_force = max(max_force, torch.max(torch.norm(forces, dim=-1)).item())
                if max_force > self.cfg.contact_force_threshold:
                    contact_detected = True

        results = {
            'contact_detected': contact_detected,
            'max_contact_force': max_force
        }

        print(f"  Contact detected: {contact_detected}")
        print(f"  Max contact force: {max_force:.2f} N")
        print("  Note: Collision filtering validation requires specific test geometry")

        return results

    def test_determinism(self) -> dict:
        """Test 6: Deterministic behavior"""
        print("\n" + "="*60)
        print("TEST 6: Determinism")
        print("="*60)

        def run_sequence(seed: int):
            """Run fixed command sequence"""
            configure_seed(42)
            self.sim.reset()

            states = []
            target = torch.zeros(1, self.num_joints, device=self.device)

            for step in range(100):
                # Deterministic command sequence
                target[0, 0] = 0.1 * np.sin(step * 0.1)
                self.robot.set_joint_position_target(target)
                self.sim.step()
                states.append(self.robot.data.joint_pos[0].cpu().numpy().copy())

            return np.array(states)

        # Run twice with same seed
        states_1 = run_sequence(42)
        states_2 = run_sequence(42)

        max_deviation = np.max(np.abs(states_1 - states_2))
        deterministic = max_deviation < 1e-6

        results = {
            'deterministic': deterministic,
            'max_deviation': max_deviation
        }

        status = "✓" if deterministic else "✗"
        print(f"  {status} Max deviation: {max_deviation:.2e}")
        print(f"  {status} Deterministic: {deterministic}")

        return results

    # ------------------------------------------------------------------------
    # CONTROL LOOP
    # ------------------------------------------------------------------------
    def run_control_loop(self, target_positions: torch.Tensor,
                         duration: float) -> dict:
        """Execute PD control loop"""
        num_steps = int(duration / (self.cfg.sim_dt * self.cfg.decimation))

        metrics = {
            'pos_error': [],
            'velocity': [],
            'torques': []
        }

        step_count = 0
        for step in range(num_steps):
            # Apply control for decimation steps
            for _ in range(self.cfg.decimation):
                self.robot.set_joint_position_target(target_positions)
                self.sim.step()
                step_count += 1

            # Log every 100 control steps (not sim steps)
            if step % 100 == 0:
                current_pos = self.robot.data.joint_pos
                current_vel = self.robot.data.joint_vel

                pos_error = (target_positions - current_pos).cpu().numpy()
                metrics['pos_error'].append(pos_error[0])
                metrics['velocity'].append(current_vel.cpu().numpy()[0])

                if step % 1000 == 0:
                    print(f"    Step {step}/{num_steps}: error={np.max(np.abs(pos_error)):.5f} rad")

        return metrics

    # ------------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------------
    # def compute_fk(self, joint_positions: np.ndarray) -> np.ndarray:
    #     """Manual FK from URDF (use pinocchio or robot_descriptions)"""
    #     # TODO: Implement or use library
    #     pass

    def visualize_metrics(self, metrics: dict, save_path: str):
        """Plot stability metrics"""
        pos_errors = np.array(metrics['pos_error'])
        velocities = np.array(metrics['velocity'])

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Position errors
        axes[0].plot(np.abs(pos_errors))
        axes[0].set_ylabel('Absolute Position Error (rad)')
        axes[0].set_title('Joint Position Errors Over Time')
        axes[0].grid(True)
        axes[0].legend([f'Joint {i}' for i in range(pos_errors.shape[1])])

        # Velocities
        axes[1].plot(np.abs(velocities))
        axes[1].set_ylabel('Absolute Velocity (rad/s)')
        axes[1].set_xlabel('Sample (every 100 control steps)')
        axes[1].set_title('Joint Velocities Over Time')
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"  ✓ Saved plot: {save_path}")
        plt.close()

    def print_summary(self, results: dict):
        """Print test summary"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)

        for test_name, result in results.items():
            print(f"\n{test_name.upper().replace('_', ' ')}:")
            if isinstance(result, dict):
                for key, value in result.items():
                    if not isinstance(value, (dict, list, np.ndarray)):
                        print(f"  {key}: {value}")

        print("\n" + "="*60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    # Setup config
    cfg = URDFValidationConfig()
    # cfg.usd_path = args_cli.usd_path
    cfg.test_duration = args_cli.duration
    cfg.output_dir = args_cli.output_dir

    # Initialize validator
    validator = URDFValidator(cfg)
    validator.setup()

    # Run all tests
    results = {}

    try:
        results['joint_limits'] = validator.test_joint_limits()
    except Exception as e:
        print(f"  ✗ Joint limits test failed: {e}")
        results['joint_limits'] = {'error': str(e)}

    try:
        results['joint_axes'] = validator.test_joint_axes()
    except Exception as e:
        print(f"  ✗ Joint axes test failed: {e}")
        results['joint_axes'] = {'error': str(e)}

    try:
        results['ee_frame'] = validator.test_ee_frame()
    except Exception as e:
        print(f"  ✗ EE frame test failed: {e}")
        results['ee_frame'] = {'error': str(e)}

    try:
        results['stability'] = validator.test_hold_pose_stability()
        validator.visualize_metrics(
            results['stability']['metrics'],
            f"{cfg.output_dir}/stability_metrics.png"
        )
    except Exception as e:
        print(f"  ✗ Stability test failed: {e}")
        results['stability'] = {'error': str(e)}

    try:
        results['collision'] = validator.test_collision_detection()
    except Exception as e:
        print(f"  ✗ Collision test failed: {e}")
        results['collision'] = {'error': str(e)}

    try:
        results['determinism'] = validator.test_determinism()
    except Exception as e:
        print(f"  ✗ Determinism test failed: {e}")
        results['determinism'] = {'error': str(e)}

    # Print summary
    validator.print_summary(results)

    # Cleanup
    simulation_app.close()


if __name__ == "__main__":
    main()