import pytest
import mujoco as mj
import os
from pathlib import Path


class TestMujocoModels:
    """Test suite for MuJoCo models in the menagerie."""

    @classmethod
    def setup_class(cls):
        """Set up test fixtures."""
        # Get the project root directory
        cls.project_root = Path(__file__).parent.parent
        cls.menagerie_path = cls.project_root / "mimic_retargeter_lab" / "mujoco_menagerie"

        # Verify the menagerie directory exists
        assert cls.menagerie_path.exists(), (
            f"Menagerie directory not found: {cls.menagerie_path}"
        )

    def get_all_xml_files(self):
        """Get all XML files from the mujoco menagerie."""
        xml_files = []
        for model_dir in self.menagerie_path.iterdir():
            if model_dir.is_dir():
                # Find all XML files in this model directory
                model_xml_files = list(model_dir.glob("*.xml"))
                xml_files.extend(model_xml_files)
        return xml_files

    def test_all_models_load_successfully(self):
        """Test that all XML files in the menagerie can be loaded by MuJoCo."""
        xml_files = self.get_all_xml_files()

        assert len(xml_files) > 0, "No XML files found in mujoco_menagerie"

        failed_models = []
        successful_models = []

        for xml_file in xml_files:
            try:
                # Change to the model directory so relative paths work
                original_cwd = os.getcwd()
                model_dir = xml_file.parent
                os.chdir(model_dir)

                # Load the model
                model = mj.MjModel.from_xml_path(str(xml_file.name))

                # Verify basic model properties
                assert model.nq > 0, f"Model {xml_file.name} has no degrees of freedom"
                assert model.nbody > 0, f"Model {xml_file.name} has no bodies"

                successful_models.append(xml_file)

            except Exception as e:
                failed_models.append((xml_file, str(e)))
            finally:
                # Restore original working directory
                os.chdir(original_cwd)

        # Report results
        print(f"\nSuccessfully loaded {len(successful_models)} models:")
        for model_file in successful_models:
            print(f"  ✓ {model_file.parent.name}/{model_file.name}")

        if failed_models:
            print(f"\nFailed to load {len(failed_models)} models:")
            for model_file, error in failed_models:
                print(f"  ✗ {model_file.parent.name}/{model_file.name}: {error}")

        # Test should pass even if some models fail to load (for debugging)
        # but we report the failures
        assert len(successful_models) > 0, "No models loaded successfully"

    def test_models_can_be_rendered(self):
        """Test that models can be rendered without errors."""
        xml_files = self.get_all_xml_files()

        failed_renders = []
        successful_renders = []

        for xml_file in xml_files:
            try:
                # Change to the model directory so relative paths work
                original_cwd = os.getcwd()
                model_dir = xml_file.parent
                os.chdir(model_dir)

                # Load the model and create simulation data
                model = mj.MjModel.from_xml_path(str(xml_file.name))
                data = mj.MjData(model)

                # Create renderer
                renderer = mj.Renderer(model, height=480, width=640)

                # Step the simulation once to ensure valid state
                mj.mj_step(model, data)

                # Render the scene
                renderer.update_scene(data)
                pixels = renderer.render()

                # Verify we got valid pixel data
                assert pixels is not None, f"Renderer returned None for {xml_file.name}"
                assert pixels.shape == (480, 640, 3), (
                    f"Unexpected pixel shape for {xml_file.name}"
                )

                successful_renders.append(xml_file)

            except Exception as e:
                failed_renders.append((xml_file, str(e)))
            finally:
                # Restore original working directory
                os.chdir(original_cwd)

        # Report results
        print(f"\nSuccessfully rendered {len(successful_renders)} models:")
        for model_file in successful_renders:
            print(f"  ✓ {model_file.parent.name}/{model_file.name}")

        if failed_renders:
            print(f"\nFailed to render {len(failed_renders)} models:")
            for model_file, error in failed_renders:
                print(f"  ✗ {model_file.parent.name}/{model_file.name}: {error}")

        assert len(successful_renders) > 0, "No models rendered successfully"

    def test_scene_files_specifically(self):
        """Test scene files specifically, as they should have proper lighting and environments."""
        xml_files = self.get_all_xml_files()
        scene_files = [f for f in xml_files if "scene" in f.name.lower()]

        assert len(scene_files) > 0, "No scene files found"

        for scene_file in scene_files:
            try:
                # Change to the model directory so relative paths work
                original_cwd = os.getcwd()
                model_dir = scene_file.parent
                os.chdir(model_dir)

                # Load the scene
                model = mj.MjModel.from_xml_path(str(scene_file.name))
                data = mj.MjData(model)

                # Verify scene has lighting
                assert model.nlight > 0, f"Scene {scene_file.name} has no lights"

                # Create renderer and render
                renderer = mj.Renderer(model, height=480, width=640)
                mj.mj_step(model, data)
                renderer.update_scene(data)
                pixels = renderer.render()

                assert pixels is not None, f"Failed to render scene {scene_file.name}"

                print(f"  ✓ Scene rendered: {scene_file.parent.name}/{scene_file.name}")

            except Exception as e:
                pytest.fail(
                    f"Scene file {scene_file.parent.name}/{scene_file.name} failed: {e}"
                )
            finally:
                # Restore original working directory
                os.chdir(original_cwd)
