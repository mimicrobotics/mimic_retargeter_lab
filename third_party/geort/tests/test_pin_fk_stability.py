import unittest
import numpy as np
import sys
import os

# Ensure we can import from the source directory
sys.path.append(os.getcwd())

from geort.env.hand_min import HandKinematicModel


class TestHandKinematicsStability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load the model once for all tests."""
        print("\n[SetUp] Loading Hand Model...")
        # Assuming you are running from repo root
        try:
            # You might need to adjust the hardcoded 'p50' or config path depending on your setup
            # For this test, we mimic what test_forward_kinematics.py does:
            cls.config = {
                "urdf_path": "./assets/p50/p50.urdf",
                "joint_order": [
                    "thumb_base2cmc",
                    "thumb_cmc2mcp",
                    "thumb_mcp2pp",
                    "thumb_pp2dp_actuated",
                    "index_base2mcp",
                    "index_mcp2pp",
                    "index_pp2mp",
                    "middle_base2mcp",
                    "middle_mcp2pp",
                    "middle_pp2mp",
                    "ring_base2mcp",
                    "ring_mcp2pp",
                    "ring_pp2mp",
                    "pinky_base2mcp",
                    "pinky_mcp2pp",
                    "pinky_pp2mp",
                ],
                "fingertip_link": [
                    {
                        "name": "thumb",
                        "link": "thumb_fingertip",
                        "center_offset": [0, 0, 0],
                    },
                    {
                        "name": "index",
                        "link": "index_fingertip",
                        "center_offset": [0, 0, 0],
                    },
                ],
            }

            cls.model = HandKinematicModel.build_from_config(cls.config)

            # Initialize keypoints (required before calling keypoint_from_qpos)
            keypoint_links = [info["link"] for info in cls.config["fingertip_link"]]
            keypoint_offsets = [
                info["center_offset"] for info in cls.config["fingertip_link"]
            ]
            cls.model.initialize_keypoint(keypoint_links, keypoint_offsets)

        except Exception as e:
            raise unittest.SkipTest(
                f"Could not load model assets (run from repo root): {e}"
            )

    def test_01_return_types_are_pure_numpy(self):
        """
        CRITICAL: Ensure returned orientation is a numpy array,
        NOT a C++ pinocchio.Quaternion object (which causes Segfaults).
        """
        qpos_zero = np.zeros(self.model.n_user_dof)
        output = self.model.keypoint_from_qpos(qpos_zero, ret_orientation=True)

        thumb_data = output["thumb_fingertip"]
        pos, quat = thumb_data

        # Check Position
        self.assertIsInstance(pos, np.ndarray, "Position must be a numpy array")
        self.assertEqual(pos.dtype, np.float64, "Position must be float64")

        # Check Quaternion (The source of the previous crash)
        self.assertIsInstance(
            quat, np.ndarray, "Quaternion must be a numpy array, NOT pin.Quaternion"
        )
        self.assertEqual(quat.shape, (4,), "Quaternion must be size 4 [x,y,z,w]")

        # Explicit check to ensure no C++ bindings leaked
        self.assertFalse(
            str(type(quat)).count("pinocchio"),
            f"Detected Pinocchio C++ object in output: {type(quat)}. This will cause crashes!",
        )

    def test_02_stress_test_stability(self):
        """
        Run FK 10,000 times with random inputs to check for memory leaks
        or alignment segmentation faults.
        """
        n_iterations = 10000
        print(f"\n[StressTest] Running FK {n_iterations} times...")

        for i in range(n_iterations):
            # Generate random joint angles between -1.0 and 1.0 rad
            q_rand = np.random.uniform(-1.0, 1.0, size=self.model.n_user_dof)

            # We just want to ensure this doesn't crash
            try:
                _ = self.model.keypoint_from_qpos(q_rand, ret_orientation=True)
            except Exception as e:
                self.fail(f"Crashed on iteration {i} with input {q_rand}: {e}")

        print(f"[StressTest] Successfully survived {n_iterations} iterations.")


if __name__ == "__main__":
    unittest.main()
