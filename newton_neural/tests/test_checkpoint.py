# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for safe checkpoint serialization and ``NeRDBase.save`` / ``load`` validation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import newton
import torch

import newton_neural
from newton_neural._src.core import checkpoint
from newton_neural.models import ContactInjection, FlashNeRD, Precision
from newton_neural.predictions import PredictionFreeMotion
from newton_neural.tests.extensions import ContactInputRootHeight
from newton_neural.tests.helpers import (
    NERDS,
    WORLD_COUNT,
    frame_of,
    make_backbone,
    make_frames,
    make_nerd,
    make_system,
)
from newton_neural.tests.unittest_utils import FolderTestCase, add_function_test


def _file(test: TestCheckpoint) -> Path:
    """A path in the scratch folder of the test, in a folder that does not exist yet."""
    return test.folder / "run" / "model.pt"


def _rewrite(path: Path, **changes: object) -> None:
    """Change entries of the manifest of a checkpoint file."""
    payload = torch.load(path, weights_only=True)
    payload["manifest"].update(changes)
    torch.save(payload, path)


def test_save_and_load(test: TestCheckpoint, device: torch.device, robot: str, contact: bool, backbone: str) -> None:
    """Save a model, load it on a device and expect the same parts, settings and predictions."""
    nerd = make_nerd(robot, device, contact_input=ContactInputRootHeight() if contact else None, **NERDS[backbone])
    nerd.precision = Precision.FP16
    nerd.metadata = {"dataset": "fixture", "held_out_worlds": [1, 5], "training": {"steps": 3}}
    path = _file(test)
    nerd.save(path)
    loaded = FlashNeRD.load(path, device=device)

    test.assertEqual(loaded.device.type, device.type)
    for part in ("system", "contact_input", "prediction"):
        test.assertIs(type(getattr(loaded, part)), type(getattr(nerd, part)))
    test.assertEqual(loaded.system.spec, nerd.system.spec)
    test.assertEqual(loaded.network.backbone.config(), nerd.network.backbone.config())
    test.assertEqual(loaded.network.readout_hidden.out_features, 16)
    test.assertEqual(loaded.signals, nerd.signals)
    test.assertEqual(loaded.frame_dt, nerd.frame_dt)
    test.assertIs(loaded.precision, Precision.FP16)
    test.assertEqual(loaded.metadata, nerd.metadata)
    test.assertEqual(loaded.state_dict().keys(), nerd.state_dict().keys())

    window = make_frames(nerd, WORLD_COUNT, 4, device=device)
    histories = nerd.history(WORLD_COUNT), loaded.history(WORLD_COUNT)
    with torch.no_grad():
        for index in range(4):
            frames = frame_of(window, index)
            test.assertTrue(torch.equal(nerd.predict(frames, histories[0]), loaded.predict(frames, histories[1])))


def test_load_defaults_to_cpu(test: TestCheckpoint, device: torch.device) -> None:
    """Save a model from a device, load it without one and expect it on the CPU."""
    path = _file(test)
    make_nerd("arm", device).save(path)
    test.assertEqual(FlashNeRD.load(path).device.type, "cpu")


class TestCheckpoint(FolderTestCase):
    def test_write_and_read(self) -> None:
        """Write a manifest and tensors, read them back and expect them stamped with format and versions."""
        path = _file(self)
        state = {"weight": torch.arange(6.0).view(2, 3), "count": torch.tensor(3, dtype=torch.int64)}
        checkpoint.write(path, {"model": "Example", "values": [1, 2.5, None, {"a": True}]}, state)
        manifest, loaded = checkpoint.read(path)
        self.assertEqual(
            manifest,
            {
                "format": checkpoint.FORMAT,
                "library": newton_neural.__version__,
                "newton": newton.__version__,
                "model": "Example",
                "values": [1, 2.5, None, {"a": True}],
            },
        )
        self.assertEqual(loaded.keys(), state.keys())
        for name, value in state.items():
            self.assertTrue(torch.equal(loaded[name], value))
            self.assertEqual(loaded[name].dtype, value.dtype)

    def test_manifest_is_plain_data(self) -> None:
        """Save a model with a contact input and expect a file of plain data that loads without running code."""
        nerd = make_nerd("quadruped", "cpu", contact_input=ContactInputRootHeight())
        path = _file(self)
        nerd.save(path)
        payload = torch.load(path, weights_only=True)
        self.assertEqual(set(payload), {"manifest", "state"})
        manifest = payload["manifest"]
        self.assertEqual(json.loads(json.dumps(manifest)), manifest)
        self.assertEqual(
            set(manifest),
            {
                "format",
                "library",
                "newton",
                "model",
                "parts",
                "settings",
                "frame_dt",
                "precision",
                "signals",
                "metadata",
            },
        )
        self.assertEqual(manifest["model"], "FlashNeRD")
        names = {kind: part["name"] for kind, part in manifest["parts"].items()}
        expected = {
            "system": "SystemArticulation",
            "contact_input": "ContactInputRootHeight",
            "backbone": "BackboneFlash",
            "prediction": "PredictionStateIncrement",
        }
        self.assertEqual(names, expected)
        self.assertEqual(manifest["parts"]["contact_input"]["config"], {"code_count": 4})
        settings = {"contact_injections": ["embedding", "readout", "increment"], "readout_hidden_dim": 16}
        self.assertEqual(manifest["settings"], settings)
        self.assertEqual(manifest["precision"], "tf32")
        self.assertEqual([signal["name"] for signal in manifest["signals"]], list(nerd.signals))
        self.assertTrue(all(isinstance(value, torch.Tensor) for value in payload["state"].values()))

    def test_model_without_contact_input(self) -> None:
        """Save a free motion model without a contact input and expect a null part and the prediction back."""
        nerd = make_nerd("arm", "cpu", prediction=PredictionFreeMotion())
        path = _file(self)
        nerd.save(path)
        manifest = checkpoint.read(path)[0]
        self.assertIsNone(manifest["parts"]["contact_input"])
        self.assertEqual(manifest["settings"], {"contact_injections": None, "readout_hidden_dim": 16})
        loaded = FlashNeRD.load(path)
        self.assertIsNone(loaded.contact_input)
        self.assertEqual(loaded.contact_injections, ())
        self.assertIsInstance(loaded.prediction, PredictionFreeMotion)

    def test_contact_injections_round_trip(self) -> None:
        """Save a model that adds the contact code at two places and expect the choice and the predictions back."""
        places = (ContactInjection.INCREMENT, ContactInjection.EMBEDDING)
        nerd = make_nerd("quadruped", "cpu", contact_input=ContactInputRootHeight(), contact_injections=places)
        path = _file(self)
        nerd.save(path)
        self.assertEqual(checkpoint.read(path)[0]["settings"]["contact_injections"], ["increment", "embedding"])
        loaded = FlashNeRD.load(path)
        self.assertEqual(loaded.contact_injections, places)
        self.assertEqual(loaded.state_dict().keys(), nerd.state_dict().keys())
        frames = make_frames(nerd, WORLD_COUNT)
        with torch.no_grad():
            expected = nerd.predict(frames, nerd.history(WORLD_COUNT))
            self.assertTrue(torch.equal(loaded.predict(frames, loaded.history(WORLD_COUNT)), expected))

    def test_fitted_follows_the_file(self) -> None:
        """Save a fitted model and one that never was and expect each loaded model to say the same."""
        path = _file(self)
        make_nerd("arm", "cpu").save(path)
        self.assertTrue(FlashNeRD.load(path).fitted)
        FlashNeRD(make_system("arm"), backbone=make_backbone()).save(path)
        self.assertFalse(FlashNeRD.load(path).fitted)

    def test_precision_as_string(self) -> None:
        """Set the precision of a model as a string, save it and expect the precision back. Refuse an unknown one."""
        nerd = make_nerd("arm", "cpu")
        nerd.precision = "bf16"
        path = _file(self)
        nerd.save(path)
        self.assertIs(FlashNeRD.load(path).precision, Precision.BF16)
        nerd.precision = "fp8"
        with self.assertRaisesRegex(ValueError, "fp8"):
            nerd.save(path)

    def test_other_formats_are_refused(self) -> None:
        """Mark a file as written with a later or an earlier format and expect a ValueError that names its writer."""
        path = _file(self)
        make_nerd("arm", "cpu").save(path)
        for written in (checkpoint.FORMAT + 1, checkpoint.FORMAT - 1):
            _rewrite(path, format=written, library="99.0.0")
            with self.assertRaisesRegex(ValueError, rf"format {written}.*99\.0\.0.*reads format {checkpoint.FORMAT}"):
                FlashNeRD.load(path)

    def test_other_files_are_refused(self) -> None:
        """Load a file that holds something else and expect a ValueError."""
        path = _file(self)
        path.parent.mkdir()
        torch.save({"weights": torch.zeros(2)}, path)
        with self.assertRaisesRegex(ValueError, "not a newton_neural checkpoint"):
            FlashNeRD.load(path)
        # A file of the same two entries whose manifest names no format is no checkpoint either.
        for manifest in ({"model": "FlashNeRD"}, {"format": "1"}, ["format"]):
            torch.save({"manifest": manifest, "state": {}}, path)
            with self.assertRaisesRegex(ValueError, "not a newton_neural checkpoint"):
                checkpoint.read(path)

    def test_changed_signals_are_refused(self) -> None:
        """Load a file whose parts give other signals than it was written for and expect the first one named."""
        path = _file(self)
        make_nerd("quadruped", "cpu", contact_input=ContactInputRootHeight()).save(path)
        signals = checkpoint.read(path)[0]["signals"]
        self.assertEqual(
            [signal["name"] for signal in signals], ["joint_q", "joint_qd", "joint_target_q", "root_height"]
        )

        wider = [dict(signal) for signal in signals]
        wider[1]["shape"], wider[3]["unit"] = [99], "mm"
        _rewrite(path, signals=wider)
        written, rebuilt = (
            r"signal 1 is \{'name': 'joint_qd', 'shape': \[99\]",
            r"give \{'name': 'joint_qd', 'shape': \[18\]",
        )
        with self.assertRaisesRegex(ValueError, f"{written}.*{rebuilt}.*Train and save the model again"):
            FlashNeRD.load(path)
        _rewrite(path, signals=signals[:-1])
        with self.assertRaisesRegex(ValueError, "signal 3 is absent.*give {'name': 'root_height'"):
            FlashNeRD.load(path)
        _rewrite(path, signals=[*signals, signals[0]])
        with self.assertRaisesRegex(ValueError, "signal 4 is {'name': 'joint_q'.*give absent"):
            FlashNeRD.load(path)
        _rewrite(path, signals=signals)
        self.assertEqual(list(FlashNeRD.load(path).signals), [signal["name"] for signal in signals])

    def test_unknown_model_class_is_named(self) -> None:
        """Load a checkpoint whose model class is not registered and expect a ValueError that names it."""
        path = _file(self)
        make_nerd("arm", "cpu").save(path)
        _rewrite(path, model="SuperMimic")
        with self.assertRaisesRegex(ValueError, "unknown NeRDBase 'SuperMimic'.*FlashNeRD.*Import the module"):
            FlashNeRD.load(path)

    def test_unknown_part_is_named(self) -> None:
        """Load a checkpoint whose backbone class is not registered and expect a ValueError that names it."""
        path = _file(self)
        make_nerd("arm", "cpu").save(path)
        parts = checkpoint.read(path)[0]["parts"]
        parts["backbone"]["name"] = "BackboneUnknown"
        _rewrite(path, parts=parts)
        with self.assertRaisesRegex(ValueError, "BackboneUnknown"):
            FlashNeRD.load(path)

    def test_metadata_must_be_plain_data(self) -> None:
        """Save a model whose metadata holds a tensor and expect a TypeError and no file."""
        nerd = make_nerd("arm", "cpu")
        nerd.metadata = {"loss": torch.tensor(0.5)}
        path = _file(self)
        with self.assertRaisesRegex(TypeError, "plain data"):
            nerd.save(path)
        self.assertFalse(path.exists())


for backbone_name in ("flash", "transformer", "mlp"):
    add_function_test(
        TestCheckpoint,
        f"test_save_and_load_quadruped_contact_{backbone_name}",
        test_save_and_load,
        robot="quadruped",
        contact=True,
        backbone=backbone_name,
    )
for robot_name in ("quadruped", "arm"):
    add_function_test(
        TestCheckpoint,
        f"test_save_and_load_{robot_name}",
        test_save_and_load,
        robot=robot_name,
        contact=False,
        backbone="flash",
    )
add_function_test(TestCheckpoint, "test_load_defaults_to_cpu", test_load_defaults_to_cpu)


if __name__ == "__main__":
    unittest.main()
