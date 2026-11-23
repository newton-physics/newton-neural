# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``TrajectoryWriter``, ``TrajectoryDataset``, and ``TrajectoryWindows``."""

from __future__ import annotations

import io
import json
import shutil
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import torch
from torch import Tensor

from newton_neural import Frames
from newton_neural._src.core.signal import SignalRole, SignalSpec
from newton_neural._src.data.dataset import _PENDING_LIMIT, TrajectoryDataset, TrajectoryWriter
from newton_neural._src.data.windows import TrajectoryBatch, TrajectoryWindows
from newton_neural.tests.unittest_utils import FolderTestCase, add_function_test

FRAME_DT = 1.0 / 64.0

SIGNALS = {
    "joint_q": SignalSpec("joint_q", (3,), SignalRole.STATE, unit="m or rad"),
    "contact_links": SignalSpec("contact_links", (2,), SignalRole.CONTACT, dtype="int32"),
    "touching": SignalSpec("touching", (), SignalRole.CONTACT, dtype="bool"),
}


def _frames(world_ids: Tensor, frame_ids: Tensor) -> Frames:
    """Signals ``[world_count, frame_count, *shape]`` whose values name their world and frame."""
    world = world_ids[:, None].expand(len(world_ids), len(frame_ids))
    frame = frame_ids[None, :].expand_as(world)
    return {
        "joint_q": torch.stack((world, frame, 1000 * world + frame), dim=-1).float(),
        "contact_links": torch.stack((world, -frame), dim=-1).int(),
        "touching": (world + frame) % 2 == 0,
    }


def _flags(world_count: int, frame_count: int) -> tuple[Tensor, Tensor]:
    """Random episode starts and invalid frames ``[world_count, frame_count]``, each world starting an episode."""
    generator = torch.Generator().manual_seed(0)
    episode_start = torch.rand(world_count, frame_count, generator=generator) < 0.08
    episode_start[:, 0] = True
    valid = torch.rand(world_count, frame_count, generator=generator) > 0.04
    return episode_start, valid


def _record(path: Path, episode_start: Tensor, valid: Tensor, *, shard_count: int = 1) -> TrajectoryDataset:
    """Write one recording with the given flags, split into shards that continue one another."""
    world_count, frame_count = valid.shape
    frames = _frames(torch.arange(world_count), torch.arange(frame_count))
    writer = TrajectoryWriter(path, SIGNALS, FRAME_DT)
    for i, columns in enumerate(torch.arange(frame_count).chunk(shard_count)):
        shard = {name: value[:, columns] for name, value in frames.items()}
        writer.write(shard, episode_start[:, columns], valid[:, columns], continues=i > 0)
    return writer.close()


def _one_episode(world_count: int, frame_count: int) -> tuple[Tensor, Tensor]:
    """Flags of worlds that each hold one episode of valid frames."""
    episode_start = torch.zeros(world_count, frame_count, dtype=torch.bool)
    episode_start[:, 0] = True
    return episode_start, torch.ones(world_count, frame_count, dtype=torch.bool)


def _world_and_frame(batch: TrajectoryBatch) -> Tensor:
    """World id and frame number of every frame of every window, ``[batch_size, length + 1, 2]`` int64."""
    return batch.frames["joint_q"][..., :2].long()


class TestTrajectoryDataset(FolderTestCase):
    def test_round_trip(self) -> None:
        """Read back the values, dtypes and metadata that were written across several shards."""
        generation = {"episode_frames": [200, 400], "world_count": 6}
        writer = TrajectoryWriter(self.folder / "data", SIGNALS, FRAME_DT, generation=generation)
        # Four worlds over two shards that continue one another, then two new worlds in a third shard.
        blocks = [
            (torch.arange(4), torch.arange(0, 6), False),
            (torch.arange(4), torch.arange(6, 16), True),
            (torch.arange(4, 6), torch.arange(0, 16), False),
        ]
        written = []
        for world_ids, frame_ids, continues in blocks:
            frames = _frames(world_ids, frame_ids)
            episode_start = frame_ids.expand(len(world_ids), -1) % 5 == 0
            valid = frame_ids.expand(len(world_ids), -1) % 7 != 3
            writer.write(frames, episode_start, valid, continues=continues)
            written.append({**frames, "episode_start": episode_start, "valid": valid})
        dataset = writer.close()

        self.assertEqual(dataset.signals, SIGNALS)
        self.assertEqual(dataset.frame_dt, FRAME_DT)
        self.assertEqual(dataset.generation, generation | {"complete": True})
        self.assertEqual(generation, {"episode_frames": [200, 400], "world_count": 6})  # the caller's stays its own
        meta = json.loads((self.folder / "data" / "meta.json").read_text())
        self.assertEqual(sorted(meta), ["format", "frame_dt", "generation", "shards", "signals"])
        self.assertEqual(dataset.world_count, 6)
        self.assertEqual(dataset.frame_count, 16)
        self.assertEqual([shard["continues"] for shard in dataset.shards], [False, True, False])
        self.assertEqual([shard["frame_count"] for shard in dataset.shards], [6, 10, 16])
        dtypes = {"joint_q": np.float32, "contact_links": np.int32, "touching": np.bool_}
        for shard, expected in enumerate(written):
            arrays = dataset.arrays(shard)
            self.assertEqual(arrays.keys(), expected.keys())
            for name, value in expected.items():
                self.assertIsInstance(arrays[name], np.memmap)
                self.assertEqual(arrays[name].dtype, dtypes.get(name, np.bool_))
                np.testing.assert_array_equal(arrays[name], value.numpy())
        # One file per signal and flag in each shard, and nothing else.
        self.assertEqual(len(list((self.folder / "data").rglob("*.npy"))), 3 * (len(SIGNALS) + 2))
        self.assertIn("96 frames of 6 worlds", repr(TrajectoryDataset(self.folder / "data")))

    def test_frame_count_is_per_world(self) -> None:
        """Expect the frames of every world, zero without shards, and an error for recordings of two lengths."""
        self.assertEqual(TrajectoryWriter(self.folder / "empty", SIGNALS, FRAME_DT).close().frame_count, 0)
        self.assertEqual(_record(self.folder / "data", *_one_episode(3, 10), shard_count=3).frame_count, 10)
        uneven = TrajectoryWriter(self.folder / "uneven", SIGNALS, FRAME_DT)
        uneven.write(_frames(torch.arange(2), torch.arange(8)), *_one_episode(2, 8))
        uneven.write(_frames(torch.arange(3), torch.arange(6)), *_one_episode(3, 6))
        dataset = uneven.close()
        with self.assertRaisesRegex(ValueError, r"same number of frames per world.*\[6, 8\]"):
            _ = dataset.frame_count
        self.assertIn("34 frames of 5 worlds", repr(dataset))  # the total exists for any recordings

    def test_write_copies_frames(self) -> None:
        """Keep the frames as they were when written, although the caller overwrites its tensors at once."""
        episode_start, valid = _one_episode(2, 4)
        frames = _frames(torch.arange(2), torch.arange(4))
        expected = {name: value.clone() for name, value in frames.items()}
        writer = TrajectoryWriter(self.folder / "data", SIGNALS, FRAME_DT)
        writer.write(frames, episode_start, valid)
        for value in frames.values():
            value.zero_()
        arrays = writer.close().arrays(0)
        for name, value in expected.items():
            np.testing.assert_array_equal(arrays[name], value.numpy())

    def test_writer_refuses_mismatches(self) -> None:
        """Refuse frames that differ from the declared signals, and folders or names that are taken."""
        episode_start, valid = _one_episode(2, 4)
        frames = _frames(torch.arange(2), torch.arange(4))
        writer = TrajectoryWriter(self.folder / "data", SIGNALS, FRAME_DT)
        with self.assertRaisesRegex(ValueError, "contact_links"):
            writer.write({**frames, "contact_links": frames["contact_links"].long()}, episode_start, valid)
        with self.assertRaisesRegex(ValueError, "joint_q"):
            writer.write({**frames, "joint_q": frames["joint_q"][..., :2]}, episode_start, valid)
        with self.assertRaisesRegex(ValueError, "the dataset records"):
            writer.write({"joint_q": frames["joint_q"]}, episode_start, valid)
        with self.assertRaisesRegex(ValueError, "valid"):
            writer.write(frames, episode_start, valid.float())
        with self.assertRaisesRegex(ValueError, "continue"):
            writer.write(frames, episode_start, valid, continues=True)
        writer.write(frames, episode_start, valid)
        with self.assertRaisesRegex(ValueError, "continue"):
            writer.write(_frames(torch.arange(3), torch.arange(4)), *_one_episode(3, 4), continues=True)
        writer.close()
        with self.assertRaises(FileExistsError):
            TrajectoryWriter(self.folder / "data", SIGNALS, FRAME_DT)
        with self.assertRaisesRegex(ValueError, "valid"):
            TrajectoryWriter(self.folder / "other", {"valid": SignalSpec("valid", (), SignalRole.STATE)}, FRAME_DT)
        with self.assertRaises(TypeError):
            TrajectoryWriter(self.folder / "other", SIGNALS, FRAME_DT, generation={"states": object()})
        self.assertFalse((self.folder / "other").exists())

    def test_signal_names_are_file_names(self) -> None:
        """Refuse a signal name that is not an identifier, which a shard would write outside its folder."""
        for name in ("../../outside", "a/b", "..", "", "joint q", "joint.q"):
            with self.assertRaisesRegex(ValueError, "identifier"):
                SignalSpec(name, (), SignalRole.STATE)
        dataset = _record(self.folder / "data", *_one_episode(2, 4))
        description = json.loads((dataset.path / "meta.json").read_text())
        description["signals"][0]["name"] = "../../outside"
        (dataset.path / "meta.json").write_text(json.dumps(description))
        with self.assertRaisesRegex(ValueError, "identifier"):
            TrajectoryDataset(dataset.path)

    def test_writer_leaves_a_dataset_after_an_error(self) -> None:
        """Leave the block of a writer by an error and open the shards written until then, marked incomplete."""
        episode_start, valid = _one_episode(2, 4)
        for error in (RuntimeError, KeyboardInterrupt):
            path = self.folder / error.__name__
            with self.assertRaises(error), TrajectoryWriter(path, SIGNALS, FRAME_DT) as writer:
                writer.write(_frames(torch.arange(2), torch.arange(4)), episode_start, valid)
                following = _frames(torch.arange(2), torch.arange(4, 8))
                writer.write(following, torch.zeros_like(episode_start), valid, continues=True)
                raise error("the simulation broke")
            dataset = TrajectoryDataset(path)
            self.assertEqual(dataset.generation, {"complete": False})
            self.assertEqual((dataset.world_count, dataset.frame_count, len(dataset.shards)), (2, 8, 2))
            np.testing.assert_array_equal(dataset.arrays(1)["joint_q"][:, :, 1], [[4, 5, 6, 7]] * 2)
            self.assertEqual(len(TrajectoryWindows(dataset, 3)), 2 * 5)

        with TrajectoryWriter(self.folder / "whole", SIGNALS, FRAME_DT, generation={"seed": 3}) as writer:
            writer.write(_frames(torch.arange(2), torch.arange(4)), episode_start, valid)
        self.assertEqual(TrajectoryDataset(self.folder / "whole").generation, {"seed": 3, "complete": True})

    def test_writer_keeps_the_shards_before_a_failed_write(self) -> None:
        """List the shards written before the one the disk refused, and raise what the disk raised."""
        episode_start, valid = _one_episode(2, 4)
        path = self.folder / "data"
        writer = TrajectoryWriter(path, SIGNALS, FRAME_DT)
        (path / "shard-00001").mkdir()  # the place of the second shard is taken, so its write fails
        for i in range(3):
            frames = _frames(torch.arange(2), torch.arange(4 * i, 4 * i + 4))
            writer.write(frames, episode_start, valid, continues=i > 0)
        with self.assertRaises(FileExistsError):
            writer.close()
        dataset = TrajectoryDataset(path)
        self.assertEqual([shard["path"] for shard in dataset.shards], ["shard-00000"])
        self.assertEqual(dataset.generation, {"complete": False})
        self.assertEqual(dataset.frame_count, 4)

    def test_writer_claims_folder_before_first_shard(self) -> None:
        """A second recorder opening the same empty folder must leave the first recorder's dataset intact."""
        path = self.folder / "data"
        frames = _frames(torch.arange(2), torch.arange(4))
        with TrajectoryWriter(path, SIGNALS, FRAME_DT) as writer:
            with self.assertRaises(FileExistsError), TrajectoryWriter(path, SIGNALS, FRAME_DT):
                pass
            self.assertFalse((path / "meta.json").exists())
            writer.write(frames, *_one_episode(2, 4))
        dataset = TrajectoryDataset(path)
        self.assertEqual(dataset.generation, {"complete": True})
        np.testing.assert_array_equal(dataset.arrays(0)["joint_q"], frames["joint_q"].numpy())
        self.assertEqual({entry.name for entry in path.iterdir()}, {"meta.json", "shard-00000"})

    def test_writer_refuses_dataset_published_during_open(self) -> None:
        """Keep the winner's manifest if it finishes while the second writer is claiming the folder."""
        path = self.folder / "data"
        with TrajectoryWriter(path, SIGNALS, FRAME_DT, generation={"seed": 3}) as first:
            touch = Path.touch

            def finish_then_claim(pending: Path, *, exist_ok: bool) -> None:
                first.close()
                touch(pending, exist_ok=exist_ok)

            with (
                mock.patch.object(Path, "touch", finish_then_claim),
                self.assertRaises(FileExistsError),
                TrajectoryWriter(path, SIGNALS, FRAME_DT),
            ):
                pass
        self.assertEqual(TrajectoryDataset(path).generation, {"seed": 3, "complete": True})
        self.assertEqual({entry.name for entry in path.iterdir()}, {"meta.json"})

    def test_foreign_file_in_the_folder(self) -> None:
        """Write into a folder that holds a file of another kind, read the dataset beside it, and leave it be."""
        path = self.folder / "data"
        path.mkdir()
        (path / "flashnerd.pt").write_bytes(b"a saved model")
        with TrajectoryWriter(path, SIGNALS, FRAME_DT) as writer:
            writer.write(_frames(torch.arange(2), torch.arange(4)), *_one_episode(2, 4))
        dataset = TrajectoryDataset(path)
        self.assertEqual((dataset.world_count, dataset.frame_count), (2, 4))
        self.assertEqual(len(TrajectoryWindows(dataset, 3)), 2)
        self.assertEqual((path / "flashnerd.pt").read_bytes(), b"a saved model")
        # What stands in the way of a second dataset is the first one, with the other file or without it.
        with self.assertRaisesRegex(FileExistsError, "holds a dataset.*new folder"):
            TrajectoryWriter(path, SIGNALS, FRAME_DT)
        (path / "flashnerd.pt").unlink()
        with self.assertRaisesRegex(FileExistsError, "holds a dataset"):
            TrajectoryWriter(path, SIGNALS, FRAME_DT)
        # Shards without meta.json, which a killed process leaves, are a dataset in the way as well.
        (path / "meta.json").unlink()
        with self.assertRaisesRegex(FileExistsError, "holds a dataset"):
            TrajectoryWriter(path, SIGNALS, FRAME_DT)

    def test_held_out_worlds(self) -> None:
        """Hold out the requested share of the worlds, the same ones each time the dataset is opened."""
        dataset = _record(self.folder / "data", *_one_episode(40, 4))
        held_out = dataset.held_out_worlds(0.25)
        self.assertEqual(held_out.shape, (40,))
        self.assertEqual(int(held_out.sum()), 10)
        self.assertTrue(torch.equal(TrajectoryDataset(self.folder / "data").held_out_worlds(0.25), held_out))
        self.assertFalse(torch.equal(dataset.held_out_worlds(0.25, seed=1), held_out))
        self.assertEqual(int(dataset.held_out_worlds(0.001).sum()), 1)
        with self.assertRaises(ValueError):
            dataset.held_out_worlds(1.0)


def _saved(value: Tensor) -> bytes:
    """The bytes of the ``.npy`` file that NumPy writes for a tensor."""
    file = io.BytesIO()
    np.save(file, value.cpu().numpy())
    return file.getvalue()


def _shards(device: torch.device) -> list[tuple[Frames, bool]]:
    """Shards of several sizes for a writer, each as its signals and flags and whether it continues the one before.

    Nine equal shards of one recording and a tenth with fewer frames, then new worlds in a larger and a smaller
    shard.
    """
    blocks = [(torch.arange(3), torch.arange(5 * i, 5 * i + 5), i > 0) for i in range(9)]
    blocks += [(torch.arange(3), torch.arange(45, 47), True)]
    blocks += [(torch.arange(3, 5), torch.arange(0, 40), False), (torch.arange(3, 5), torch.arange(40, 47), True)]
    shards = []
    for world_ids, frame_ids, continues in blocks:
        episode_start, valid = _flags(len(world_ids), len(frame_ids))
        arrays = _frames(world_ids, frame_ids) | {"episode_start": episode_start, "valid": valid}
        shards.append(({name: value.to(device) for name, value in arrays.items()}, continues))
    return shards


def _write(writer: TrajectoryWriter, shards: list[tuple[Frames, bool]]) -> None:
    """Write shards from tensors that are overwritten as soon as the writer has them, as a caller may."""
    for arrays, continues in shards:
        given = {name: value.clone() for name, value in arrays.items()}
        flags = given.pop("episode_start"), given.pop("valid")
        writer.write(given, *flags, continues=continues)
        for value in (*given.values(), *flags):
            value.zero_()


def _slow_disk() -> Any:
    """Replace ``np.save`` by one slower than the writes, so that shards wait and their memory is used again."""
    save = np.save

    def slow_save(file: Path, array: np.ndarray) -> None:
        time.sleep(0.002)
        save(file, array)

    return mock.patch.object(np, "save", slow_save)


def test_writer_stores_the_frames_byte_for_byte(test: TestTrajectoryDataset, device: torch.device) -> None:
    """Write more shards than may wait for a slow disk, of several sizes, and expect every file to hold the bytes
    NumPy saves for the frames as they were given. Expect the same files of a recording that is interrupted and
    of a writer that is closed as incomplete."""
    shards = _shards(device)
    expected = [{name: _saved(value) for name, value in arrays.items()} for arrays, _ in shards]

    def assert_files(path: Path, shard_count: int, complete: bool) -> None:
        meta = {
            "format": 1,
            "frame_dt": FRAME_DT,
            "signals": [spec.to_dict() for spec in SIGNALS.values()],
            "generation": {"seed": 3, "complete": complete},
            "shards": [
                {
                    "path": f"shard-{index:05d}",
                    "world_count": arrays["valid"].shape[0],
                    "frame_count": arrays["valid"].shape[1],
                    "continues": continues,
                }
                for index, (arrays, continues) in enumerate(shards[:shard_count])
            ],
        }
        test.assertEqual((path / "meta.json").read_text(), json.dumps(meta, indent=2))
        files = {str(file.relative_to(path)): file.read_bytes() for file in path.rglob("*.npy")}
        stored = {
            f"shard-{index:05d}/{name}.npy": data
            for index in range(shard_count)
            for name, data in expected[index].items()
        }
        test.assertEqual(files, stored)

    generation = {"seed": 3}
    # Staging must use host memory even when the caller's default device is CUDA.
    with _slow_disk(), torch.device(device):
        with TrajectoryWriter(test.folder / "whole", SIGNALS, FRAME_DT, generation=generation) as whole:
            _write(whole, shards)
        interrupted = TrajectoryWriter(test.folder / "interrupted", SIGNALS, FRAME_DT, generation=generation)
        with test.assertRaises(KeyboardInterrupt), interrupted:
            _write(interrupted, shards[:7])
            raise KeyboardInterrupt
        closed = TrajectoryWriter(test.folder / "closed", SIGNALS, FRAME_DT, generation=generation)
        _write(closed, shards[:7])
        closed.close(complete=False)
    assert_files(test.folder / "whole", len(shards), True)
    assert_files(test.folder / "interrupted", 7, False)
    assert_files(test.folder / "closed", 7, False)


def test_writer_reuses_its_host_memory(test: TestTrajectoryDataset, device: torch.device) -> None:
    """Write more shards than may wait for a slow disk and expect them copied into no more sets of host memory
    than shards may wait, which the writer gives back when it is closed."""
    shards = _shards(device)
    with _slow_disk(), TrajectoryWriter(test.folder / "data", SIGNALS, FRAME_DT) as writer:
        with mock.patch.object(torch, "zeros", wraps=torch.zeros) as zeros:
            _write(writer, shards[:10])
        # The writer took host memory of its own, and the ten shards went through the memory of at most four.
        array_count = len(shards[0][0])
        test.assertIn(zeros.call_count, range(array_count, _PENDING_LIMIT * array_count + 1))
        _write(writer, shards[10:])
    test.assertEqual(writer._buffers, [])


for _function in (test_writer_stores_the_frames_byte_for_byte, test_writer_reuses_its_host_memory):
    add_function_test(TestTrajectoryDataset, _function.__name__, _function)


class TestTrajectoryWindows(FolderTestCase):
    pass


def test_windows_stay_inside_episodes(test: TestTrajectoryWindows, device: torch.device) -> None:
    """Yield every run of consecutive valid frames inside one episode exactly once, and nothing else."""
    length = 4
    episode_start, valid = _flags(6, 60)
    dataset = _record(test.folder / "data", episode_start, valid, shard_count=3)
    windows = TrajectoryWindows(dataset, length, device=device)

    found = windows.measure(_world_and_frame, batch_size=16).cpu()
    test.assertEqual(found.shape, (len(windows), length + 1, 2))
    test.assertTrue((found[:, :, 0] == found[:, :1, 0]).all())
    test.assertTrue((found[:, :, 1] == found[:, :1, 1] + torch.arange(length + 1)).all())
    world, first = found[:, 0, 0], found[:, 0, 1]
    expected = {
        (w, t)
        for w in range(6)
        for t in range(60 - length)
        if valid[w, t : t + length + 1].all() and not episode_start[w, t + 1 : t + length + 1].any()
    }
    test.assertGreater(len(expected), 0)
    test.assertEqual(len(windows), len(expected))
    test.assertEqual(set(zip(world.tolist(), first.tolist(), strict=True)), expected)
    begins = windows.measure(lambda batch: batch.episode_start).cpu()
    test.assertTrue(torch.equal(begins, episode_start[world, first]))
    test.assertTrue(begins.any() and not begins.all())
    test.assertTrue(torch.equal(windows.episode_start.cpu(), begins))
    test.assertEqual(windows.episode_start.device, device)
    test.assertEqual(len(windows.with_length(60).episode_start), 0)


def test_single_frame_windows(test: TestTrajectoryWindows, device: torch.device) -> None:
    """Yield every valid frame as a window of its own when windows hold no step."""
    episode_start, valid = _flags(6, 60)
    dataset = _record(test.folder / "data", episode_start, valid, shard_count=3)
    windows = TrajectoryWindows(dataset, 0, device=device)
    test.assertEqual(len(windows), int(valid.sum()))

    batch = windows.batch(torch.arange(len(windows), device=device))
    for name, spec in SIGNALS.items():
        test.assertEqual(batch.frames[name].shape, (len(windows), 1, *spec.shape))
    world, frame = _world_and_frame(batch)[:, 0].cpu().unbind(-1)
    test.assertEqual(
        set(zip(world.tolist(), frame.tolist(), strict=True)), {(w, t) for w, t in valid.nonzero().tolist()}
    )
    test.assertTrue(torch.equal(batch.episode_start.cpu(), episode_start[world, frame]))
    test.assertEqual(len(TrajectoryWindows(dataset, 4, device=device).with_length(0)), len(windows))

    # Independent samples, where every frame starts an episode, are all single frame windows and nothing longer.
    samples = _record(test.folder / "samples", torch.ones(3, 8, dtype=torch.bool), torch.ones(3, 8, dtype=torch.bool))
    test.assertEqual(len(TrajectoryWindows(samples, 0, device=device)), 3 * 8)
    longer = TrajectoryWindows(samples, 1, device=device)
    test.assertEqual(len(longer), 0)
    with test.assertRaisesRegex(ValueError, "no window of 1 steps.*none of the 3 worlds.*2 valid.*episode_frames"):
        longer.loader(1)


def test_windows_span_continued_shards(test: TestTrajectoryWindows, device: torch.device) -> None:
    """Join shards that continue one another into one stretch of frames, and keep recordings apart."""
    writer = TrajectoryWriter(test.folder / "data", SIGNALS, FRAME_DT)
    for world_ids in (torch.arange(3), torch.arange(3, 5)):
        for i, frame_ids in enumerate(torch.arange(40).chunk(4)):
            flags = _one_episode(len(world_ids), 40)
            writer.write(_frames(world_ids, frame_ids), flags[0][:, frame_ids], flags[1][:, frame_ids], continues=i > 0)
    windows = TrajectoryWindows(writer.close(), 25, device=device)  # longer than two shards
    test.assertEqual(len(windows), 5 * (40 - 25))
    found = windows.measure(_world_and_frame).cpu()
    test.assertTrue((found[:, :, 0] == found[:, :1, 0]).all())
    test.assertTrue((found[:, :, 1] == found[:, :1, 1] + torch.arange(26)).all())
    test.assertEqual(found[:, 0, 0].unique().tolist(), [0, 1, 2, 3, 4])

    uneven = TrajectoryWriter(test.folder / "uneven", SIGNALS, FRAME_DT)
    uneven.write(_frames(torch.arange(2), torch.arange(8)), *_one_episode(2, 8))
    uneven.write(_frames(torch.arange(2), torch.arange(6)), *_one_episode(2, 6))
    with test.assertRaisesRegex(ValueError, "same number of frames"):
        TrajectoryWindows(uneven.close(), 2, device=device)


def test_loader_batches(test: TestTrajectoryWindows, device: torch.device) -> None:
    """Yield batches of the declared shapes and dtypes on the device, and drop the partial batch."""
    length, batch_size = 5, 16
    dataset = _record(test.folder / "data", *_flags(6, 60), shard_count=2)
    windows = TrajectoryWindows(test.folder / "data", length, device=device)
    test.assertGreater(len(windows) % batch_size, 0)
    full = len(windows) // batch_size

    batches = list(windows.loader(batch_size, epochs=2, generator=torch.Generator(device).manual_seed(0)))
    test.assertEqual(len(batches), 2 * full)
    for batch in batches:
        test.assertIsInstance(batch, TrajectoryBatch)
        test.assertEqual(batch.frames.keys(), SIGNALS.keys())
        for name, spec in dataset.signals.items():
            test.assertEqual(batch.frames[name].shape, (batch_size, length + 1, *spec.shape))
            test.assertEqual(batch.frames[name].dtype, spec.torch_dtype)
            test.assertEqual(batch.frames[name].device, device)
        test.assertEqual(batch.episode_start.shape, (batch_size,))
        test.assertEqual(batch.episode_start.dtype, torch.bool)
        test.assertEqual(batch.frame_dt, FRAME_DT)
        # The signals of a window belong to the same frames.
        world, frame = _world_and_frame(batch).unbind(-1)
        test.assertTrue(torch.equal(batch.frames["contact_links"], torch.stack((world, -frame), dim=-1).int()))
        test.assertTrue(torch.equal(batch.frames["touching"], (world + frame) % 2 == 0))

    # A shuffled pass shows no window twice. An unshuffled pass follows the order of the windows.
    first = torch.cat([_world_and_frame(batch)[:, 0] for batch in batches[:full]])
    test.assertEqual(len(first.unique(dim=0)), full * batch_size)
    every = windows.measure(_world_and_frame)
    ordered = torch.cat([_world_and_frame(batch) for batch in windows.loader(batch_size, shuffle=False)])
    test.assertTrue(torch.equal(ordered, every[: full * batch_size]))
    test.assertFalse(torch.equal(first, every[: full * batch_size, 0]))

    endless = windows.loader(batch_size, epochs=None)
    test.assertEqual(sum(1 for _ in zip(range(3 * full), endless, strict=False)), 3 * full)
    moved = batches[0].to("cpu")
    test.assertEqual(moved.frames["joint_q"].device.type, "cpu")
    test.assertTrue(torch.equal(moved.episode_start, batches[0].episode_start.cpu()))
    with test.assertRaisesRegex(ValueError, "batch_size"):
        windows.loader(len(windows) + 1)


def test_split_by_global_world(test: TestTrajectoryWindows, device: torch.device) -> None:
    """Hold out the same worlds in a full load and in every slice of the worlds, apart from training."""
    world_count, rank_count, length = 32, 4, 4
    dataset = _record(test.folder / "data", *_one_episode(world_count, 12), shard_count=2)
    held_out_ids = set(torch.nonzero(dataset.held_out_worlds(0.25)).flatten().tolist())
    test.assertEqual(len(held_out_ids), 8)

    def worlds_of(windows: TrajectoryWindows) -> set[int]:
        """Worlds the windows are really made of, read from the frames."""
        return set(windows.measure(lambda batch: _world_and_frame(batch)[:, 0, 0]).tolist())

    full = TrajectoryWindows(dataset, length, device=device)
    train, held_out = full.split(0.25)
    test.assertEqual(worlds_of(held_out), held_out_ids)
    test.assertEqual(worlds_of(train), set(range(world_count)) - held_out_ids)
    test.assertEqual(set(held_out.world_ids.tolist()), held_out_ids)
    test.assertEqual(len(train) + len(held_out), len(full))

    in_slices: set[int] = set()
    for rank in range(rank_count):
        part = TrajectoryWindows(dataset, length, device=device, worlds=slice(rank, None, rank_count))
        own_ids = set(range(rank, world_count, rank_count))
        test.assertEqual(part.world_ids.tolist(), sorted(own_ids))
        part_train, part_held_out = part.split(0.25)
        test.assertEqual(worlds_of(part_held_out), held_out_ids & own_ids)
        test.assertEqual(worlds_of(part_train), own_ids - held_out_ids)
        test.assertEqual(len(part_train) + len(part_held_out), len(part))
        in_slices |= worlds_of(part_held_out)
    test.assertEqual(in_slices, held_out_ids)

    # Another seed holds out other worlds, and a split of a split keeps to its own worlds.
    test.assertNotEqual(worlds_of(full.split(0.25, seed=1)[1]), held_out_ids)
    test.assertEqual(worlds_of(train.split(0.25)[0]), worlds_of(train))
    test.assertEqual(len(train.split(0.25)[1]), 0)


def test_with_length_shares_data(test: TestTrajectoryWindows, device: torch.device) -> None:
    """Give windows of another length over the same worlds without reading the dataset again."""
    dataset = _record(test.folder / "data", *_one_episode(16, 20), shard_count=2)
    windows = TrajectoryWindows(dataset, 4, device=device)
    _train, held_out = windows.split(0.25)
    shutil.rmtree(test.folder / "data")  # nothing is left to load from

    longer = windows.with_length(12)
    test.assertEqual((windows.length, longer.length), (4, 12))
    test.assertEqual((len(windows), len(longer)), (16 * 16, 16 * 8))
    batch = longer.batch(torch.arange(len(longer)))
    test.assertEqual(batch.frames["joint_q"].shape, (16 * 8, 13, 3))
    frame = _world_and_frame(batch)[:, :, 1].cpu()
    test.assertTrue((frame == frame[:, :1] + torch.arange(13)).all())

    held_out_longer = held_out.with_length(12)
    test.assertTrue(torch.equal(held_out_longer.world_ids, held_out.world_ids))
    worlds = held_out_longer.measure(lambda batch: _world_and_frame(batch)[:, 0, 0])
    test.assertEqual(worlds.unique().tolist(), held_out.world_ids.tolist())
    test.assertEqual(len(windows.with_length(20)), 0)
    with test.assertRaises(ValueError):
        windows.with_length(-1)


def test_weighted_loader(test: TestTrajectoryWindows, device: torch.device) -> None:
    """Draw windows in proportion to their weights, and never a window of zero weight."""
    dataset = _record(test.folder / "data", *_one_episode(8, 20))
    windows = TrajectoryWindows(dataset, 3, device=device)
    batch_size = 32
    weights = torch.zeros(len(windows))
    weights[[5, 77]] = torch.tensor([3.0, 1.0])
    targets = _world_and_frame(windows.batch(torch.tensor([5, 77])))[:, 0]

    generator = torch.Generator(device).manual_seed(0)
    batches = list(windows.loader(batch_size, epochs=30, generator=generator, weights=weights))
    test.assertEqual(len(batches), 30 * (len(windows) // batch_size))
    drawn = torch.cat([_world_and_frame(batch)[:, 0] for batch in batches])
    is_heavy, is_light = ((drawn == target).all(dim=1) for target in targets)
    test.assertTrue((is_heavy | is_light).all())
    test.assertAlmostEqual(is_heavy.float().mean().item(), 0.75, delta=0.04)

    with test.assertRaisesRegex(ValueError, "order"):
        windows.loader(batch_size, shuffle=False, weights=weights)
    with test.assertRaisesRegex(ValueError, "one per window"):
        windows.loader(batch_size, weights=weights[:-1])
    with test.assertRaisesRegex(ValueError, "non-negative"):
        windows.loader(batch_size, weights=torch.zeros(len(windows)))
    with test.assertRaisesRegex(ValueError, "non-negative"):
        windows.loader(batch_size, weights=weights - 1.0)


add_function_test(TestTrajectoryWindows, "test_windows_stay_inside_episodes", test_windows_stay_inside_episodes)
add_function_test(TestTrajectoryWindows, "test_single_frame_windows", test_single_frame_windows)
add_function_test(TestTrajectoryWindows, "test_windows_span_continued_shards", test_windows_span_continued_shards)
add_function_test(TestTrajectoryWindows, "test_loader_batches", test_loader_batches)
add_function_test(TestTrajectoryWindows, "test_split_by_global_world", test_split_by_global_world)
add_function_test(TestTrajectoryWindows, "test_with_length_shares_data", test_with_length_shares_data)
add_function_test(TestTrajectoryWindows, "test_weighted_loader", test_weighted_loader)


if __name__ == "__main__":
    unittest.main()
