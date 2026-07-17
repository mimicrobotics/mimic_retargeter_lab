import numpy as np
import pytest

from mimic_retargeter_lab.types import HandLandmark


class BaseHandModelRegressionSuite:
    """
    Reusable regression suite for robot hand models.

    Subclasses must provide:
      - model fixture: returns instantiated hand model
      - golden fixture: returns GoldenData for this model/chirality
      - class attrs:
          EXPECTED_QPOS_DOFS
          EXPECTED_ACTUATED_DOFS
          EXPECTED_FINGERTIPS
          EXPECTED_LANDMARK_ORDER (list[HandLandmark])
    """

    EXPECTED_QPOS_DOFS: int
    EXPECTED_ACTUATED_DOFS: int
    EXPECTED_FINGERTIPS: int
    EXPECTED_LANDMARK_ORDER: list[HandLandmark]
    # Tuples are (dependent_qpos_joint/child_joint, driver_actuated_joint/parent_joint, coefficient).
    # This matches joint_map indexing: rows=qpos joints, cols=actuated joints.
    EXPECTED_JOINT_MAP_SEMANTIC_COUPLINGS: list[tuple[str, str, float]]

    def test_fingertip_landmark_order(self, golden):
        expected = [str(lm) for lm in self.EXPECTED_LANDMARK_ORDER]
        assert golden.fingertip_landmarks == expected

    # ---------- Structure ----------

    def test_num_qpos_dofs(self, model):
        assert model.num_qpos_dofs == self.EXPECTED_QPOS_DOFS

    def test_num_actuated_dofs(self, model):
        assert model.num_actuated_dofs == self.EXPECTED_ACTUATED_DOFS

    def test_num_fingertips(self, model):
        assert model.get_num_fingertips() == self.EXPECTED_FINGERTIPS

    def test_qpos_names_length(self, model):
        assert len(model.get_qpos_joint_names()) == model.num_qpos_dofs

    def test_actuated_names_length(self, model):
        assert len(model.get_actuated_joint_names()) == model.num_actuated_dofs

    def test_qpos_names_unique(self, model):
        names = model.get_qpos_joint_names()
        assert len(names) == len(set(names))

    def test_actuated_names_unique(self, model):
        names = model.get_actuated_joint_names()
        assert len(names) == len(set(names))

    def test_joint_map_shape(self, model):
        assert model.joint_map.shape == (model.num_qpos_dofs, model.num_actuated_dofs)

    def test_joint_map_name_order_matches_golden(self, model, golden):
        assert golden.qpos_joint_names == model.get_qpos_joint_names()
        assert golden.actuated_joint_names == model.get_actuated_joint_names()

    def test_joint_map_matches_golden_exact(self, model, golden):
        jm = np.asarray(model.joint_map, dtype=np.float32)
        np.testing.assert_allclose(
            jm,
            golden.joint_map.astype(np.float32),
            atol=0.0,
            rtol=0.0,
            err_msg="Joint map differs from golden snapshot.",
        )

    def test_joint_map_row_sparsity(self, model):
        jm = np.asarray(model.joint_map, dtype=np.float32)
        row_nnz = np.count_nonzero(np.abs(jm) > 1e-8, axis=1)
        # Current coupling design maps each qpos joint to exactly one actuator dependency.
        assert np.all(row_nnz == 1), (
            f"Expected one nonzero per row, got {row_nnz.tolist()}"
        )

    def test_joint_map_semantic_couplings(self, model):
        qpos_names = model.get_qpos_joint_names()
        actuated_names = model.get_actuated_joint_names()
        qidx = {name: i for i, name in enumerate(qpos_names)}
        aidx = {name: i for i, name in enumerate(actuated_names)}
        jm = np.asarray(model.joint_map, dtype=np.float32)

        for qname, aname, coeff in self.EXPECTED_JOINT_MAP_SEMANTIC_COUPLINGS:
            assert qname in qidx, f"Missing qpos joint in model: {qname}"
            assert aname in aidx, f"Missing actuated joint in model: {aname}"
            qi = qidx[qname]
            ai = aidx[aname]
            assert jm[qi, ai] == pytest.approx(coeff), (
                f"Coupling mismatch for ({qname} <- {aname})"
            )
            # Ensure this qpos row depends only on the declared actuator.
            row_nz = np.where(np.abs(jm[qi]) > 1e-8)[0]
            assert len(row_nz) == 1 and row_nz[0] == ai, (
                f"Expected row '{qname}' to have only actuator '{aname}', "
                f"got indices {row_nz.tolist()}"
            )

    # ---------- Kinematics helpers ----------

    def _check_pose(self, model, golden, label: str):
        ctrl = golden.ctrl(label)[np.newaxis]  # (1, A)

        # Fingertip positions via MJX FK
        tip_pos = model.mjx_fk_body_positions(ctrl, joint_space="ctrl")
        tip_frames_4x4 = []
        for lm in self.EXPECTED_LANDMARK_ORDER:
            link_name = model._landmark_config[lm][0]
            pos = np.asarray(tip_pos[link_name]).squeeze(0)
            frame = np.eye(4, dtype=np.float32)
            frame[:3, 3] = pos
            tip_frames_4x4.append(frame)
        tips = np.stack(tip_frames_4x4)
        np.testing.assert_allclose(
            tips,
            golden.fingertips(label),
            atol=1e-6,
            err_msg=f"Fingertips mismatch for '{label}' pose",
        )

        kv = model.compute_keyvectors_jax(ctrl, joint_space="ctrl")
        kv_keys = sorted(kv.keys())
        kv_arr = np.stack([np.asarray(kv[k]).squeeze(0) for k in kv_keys])

        np.testing.assert_allclose(
            kv_arr,
            golden.keyvectors(label),
            atol=1e-6,
            err_msg=f"Keyvectors mismatch for '{label}' pose",
        )

    # ---------- Kinematics tests ----------

    def test_zero_ctrl_pose(self, model, golden):
        self._check_pose(model, golden, "zero")

    def test_neutral_ctrl_pose(self, model, golden):
        self._check_pose(model, golden, "neutral")

    def test_min_limits_pose(self, model, golden):
        self._check_pose(model, golden, "min_limits")

    def test_max_limits_pose(self, model, golden):
        self._check_pose(model, golden, "max_limits")

    def test_midrange_pose(self, model, golden):
        self._check_pose(model, golden, "midrange")

    def test_random_poses(self, model, golden):
        for label in golden.random_labels:
            self._check_pose(model, golden, label)

    def test_batched_poses(self, model, golden):
        labels = golden.random_labels
        ctrl_batch = np.stack([golden.ctrl(label) for label in labels])

        # Batched MJX FK: returns dict of (B, 3) positions
        tip_pos = model.mjx_fk_body_positions(ctrl_batch, joint_space="ctrl")
        tips_batch = []
        for b in range(ctrl_batch.shape[0]):
            tip_frames_4x4 = []
            for lm in self.EXPECTED_LANDMARK_ORDER:
                link_name = model._landmark_config[lm][0]
                pos = np.asarray(tip_pos[link_name])[b]
                frame = np.eye(4, dtype=np.float32)
                frame[:3, 3] = pos
                tip_frames_4x4.append(frame)
            tips_batch.append(np.stack(tip_frames_4x4))
        tips_batch = np.stack(tips_batch)
        golden_tips = np.stack([golden.fingertips(label) for label in labels])

        np.testing.assert_allclose(
            tips_batch,
            golden_tips,
            atol=1e-6,
            err_msg="Batched fingertips computation failed",
        )

    # ---------- Jacobians ----------

    def test_fingertip_jacobians(self, model, golden):
        test_labels = ["zero", "neutral", "midrange"] + golden.random_labels

        for label in test_labels:
            ctrl = golden.ctrl(label)[np.newaxis]  # (1, A)

            J_dict = model.compute_fingertip_jacobians(ctrl)

            J_np = np.stack(
                [np.asarray(J_dict[lm]) for lm in self.EXPECTED_LANDMARK_ORDER]
            )

            golden_J = golden.fingertip_jacobians(label)
            np.testing.assert_allclose(
                J_np,
                golden_J,
                atol=1e-5,
                rtol=1e-5,
                err_msg=f"Fingertip Jacobian mismatch for '{label}' pose",
            )
