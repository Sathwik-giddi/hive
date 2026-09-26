"""Tests for CheckpointStore - checkpoint persistence, indexing, and pruning.

Covers the save/load/list/delete/prune surface plus the two invariants that
keep the index consistent with what is actually on disk:

* ``checkpoint_id`` is unique, so two checkpoints taken in the same second for
  the same node cannot overwrite one another.
* ``total_checkpoints`` equals the number of distinct ids, which equals the
  number of ``cp_*.json`` files present.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from framework.schemas.checkpoint import Checkpoint, CheckpointIndex, CheckpointSummary
from framework.storage.checkpoint_store import CheckpointStore


def make_checkpoint(
    node: str = "node_a",
    checkpoint_type: str = "node_start",
    session_id: str = "sess_1",
    data_buffer: dict | None = None,
    execution_path: list[str] | None = None,
    created_at: str | None = None,
    checkpoint_id: str | None = None,
) -> Checkpoint:
    """Build a Checkpoint, optionally overriding generated fields."""
    checkpoint = Checkpoint.create(
        checkpoint_type=checkpoint_type,
        session_id=session_id,
        run_id="run_1",
        current_node=node,
        execution_path=execution_path or [],
        data_buffer=data_buffer if data_buffer is not None else {"step": 1},
    )
    overrides = {}
    if created_at is not None:
        overrides["created_at"] = created_at
    if checkpoint_id is not None:
        overrides["checkpoint_id"] = checkpoint_id
    return checkpoint.model_copy(update=overrides) if overrides else checkpoint


def checkpoint_files(store: CheckpointStore) -> list[Path]:
    """List the individual checkpoint files (excluding the index)."""
    if not store.checkpoints_dir.exists():
        return []
    return sorted(p for p in store.checkpoints_dir.glob("cp_*.json"))


# === ID UNIQUENESS ===


def test_checkpoint_ids_are_unique_within_the_same_second() -> None:
    """Regression: ids used whole-second resolution, so same-second writes collided.

    The store writes one file per id, so a collision silently discarded the
    earlier checkpoint and left the index describing a file that held
    different state.
    """
    ids = {make_checkpoint().checkpoint_id for _ in range(200)}
    assert len(ids) == 200, "checkpoint_id must be unique across rapid successive creates"


@pytest.mark.asyncio
async def test_same_second_checkpoints_do_not_overwrite_each_other(tmp_path: Path) -> None:
    """Regression: the earlier checkpoint must stay loadable after a later one."""
    store = CheckpointStore(tmp_path)
    first = make_checkpoint(data_buffer={"step": 1})
    second = make_checkpoint(data_buffer={"step": 2})
    assert first.checkpoint_id != second.checkpoint_id

    await store.save_checkpoint(first)
    await store.save_checkpoint(second)

    loaded_first = await store.load_checkpoint(first.checkpoint_id)
    assert loaded_first is not None
    assert loaded_first.data_buffer == {"step": 1}, "first checkpoint was overwritten"

    loaded_second = await store.load_checkpoint(second.checkpoint_id)
    assert loaded_second is not None
    assert loaded_second.data_buffer == {"step": 2}


@pytest.mark.asyncio
async def test_index_never_lists_the_same_id_twice(tmp_path: Path) -> None:
    """Re-saving the same id replaces its summary rather than appending a duplicate."""
    store = CheckpointStore(tmp_path)
    checkpoint = make_checkpoint(data_buffer={"step": 1})

    await store.save_checkpoint(checkpoint)
    await store.save_checkpoint(checkpoint.model_copy(update={"data_buffer": {"step": 99}}))

    index = await store.load_index()
    assert index is not None
    ids = [s.checkpoint_id for s in index.checkpoints]
    assert len(ids) == len(set(ids)) == 1
    assert index.total_checkpoints == 1


def test_index_add_checkpoint_is_idempotent() -> None:
    """CheckpointIndex.add_checkpoint replaces on duplicate id."""
    index = CheckpointIndex(session_id="sess_1")
    checkpoint = make_checkpoint()

    index.add_checkpoint(checkpoint)
    index.add_checkpoint(checkpoint)

    assert index.total_checkpoints == 1
    assert len(index.checkpoints) == 1
    assert index.latest_checkpoint_id == checkpoint.checkpoint_id


# === INDEX / DISK CONSISTENCY ===


@pytest.mark.asyncio
async def test_index_totals_match_files_on_disk(tmp_path: Path) -> None:
    """Every indexed checkpoint has exactly one file behind it."""
    store = CheckpointStore(tmp_path)
    for i in range(5):
        await store.save_checkpoint(make_checkpoint(data_buffer={"step": i}))

    index = await store.load_index()
    assert index is not None
    assert index.total_checkpoints == 5
    assert len(index.checkpoints) == 5
    assert len(checkpoint_files(store)) == 5


@pytest.mark.asyncio
async def test_concurrent_saves_keep_index_consistent(tmp_path: Path) -> None:
    """Concurrent writers must not lose or duplicate index entries."""
    import asyncio

    store = CheckpointStore(tmp_path)
    checkpoints = [make_checkpoint(data_buffer={"step": i}) for i in range(12)]
    assert len({c.checkpoint_id for c in checkpoints}) == 12

    await asyncio.gather(*(store.save_checkpoint(c) for c in checkpoints))

    index = await store.load_index()
    assert index is not None
    assert index.total_checkpoints == 12
    assert len(checkpoint_files(store)) == 12
    for checkpoint in checkpoints:
        assert await store.checkpoint_exists(checkpoint.checkpoint_id)


# === SAVE / LOAD ===


@pytest.mark.asyncio
async def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    checkpoint = make_checkpoint(data_buffer={"answer": 42}, execution_path=["n1", "n2"])

    await store.save_checkpoint(checkpoint)
    loaded = await store.load_checkpoint(checkpoint.checkpoint_id)

    assert loaded is not None
    assert loaded.data_buffer == {"answer": 42}
    assert loaded.execution_path == ["n1", "n2"]
    assert loaded.session_id == checkpoint.session_id


@pytest.mark.asyncio
async def test_load_without_id_returns_latest(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    await store.save_checkpoint(make_checkpoint(data_buffer={"step": 1}))
    latest = make_checkpoint(data_buffer={"step": 2})
    await store.save_checkpoint(latest)

    loaded = await store.load_checkpoint()
    assert loaded is not None
    assert loaded.checkpoint_id == latest.checkpoint_id
    assert loaded.data_buffer == {"step": 2}


@pytest.mark.asyncio
async def test_load_missing_checkpoint_returns_none(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    assert await store.load_checkpoint("cp_does_not_exist") is None


@pytest.mark.asyncio
async def test_load_latest_on_empty_store_returns_none(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    assert await store.load_checkpoint() is None


@pytest.mark.asyncio
async def test_corrupt_checkpoint_file_returns_none(tmp_path: Path) -> None:
    """A truncated file must not raise out of load_checkpoint."""
    store = CheckpointStore(tmp_path)
    await store.save_checkpoint(make_checkpoint())

    corrupt_id = "cp_node_start_corrupt"
    store.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    (store.checkpoints_dir / f"{corrupt_id}.json").write_text("{not json")

    assert await store.load_checkpoint(corrupt_id) is None


@pytest.mark.asyncio
async def test_checkpoint_exists(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    checkpoint = make_checkpoint()

    assert await store.checkpoint_exists(checkpoint.checkpoint_id) is False
    await store.save_checkpoint(checkpoint)
    assert await store.checkpoint_exists(checkpoint.checkpoint_id) is True


# === LISTING ===


@pytest.mark.asyncio
async def test_list_checkpoints_filters(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    await store.save_checkpoint(make_checkpoint(checkpoint_type="node_start", data_buffer={"a": 1}))
    await store.save_checkpoint(make_checkpoint(checkpoint_type="node_complete", data_buffer={"b": 2}))
    dirty = make_checkpoint(checkpoint_type="node_complete", data_buffer={"c": 3})
    await store.save_checkpoint(dirty.model_copy(update={"is_clean": False}))

    assert len(await store.list_checkpoints()) == 3
    assert len(await store.list_checkpoints(checkpoint_type="node_start")) == 1
    assert len(await store.list_checkpoints(checkpoint_type="node_complete")) == 2
    assert len(await store.list_checkpoints(is_clean=False)) == 1
    assert len(await store.list_checkpoints(is_clean=True)) == 2


@pytest.mark.asyncio
async def test_list_checkpoints_on_empty_store(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    assert await store.list_checkpoints() == []


# === DELETE ===


@pytest.mark.asyncio
async def test_delete_removes_file_and_index_entry(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    first = make_checkpoint(data_buffer={"step": 1})
    second = make_checkpoint(data_buffer={"step": 2})
    await store.save_checkpoint(first)
    await store.save_checkpoint(second)

    assert await store.delete_checkpoint(first.checkpoint_id) is True

    index = await store.load_index()
    assert index is not None
    assert [s.checkpoint_id for s in index.checkpoints] == [second.checkpoint_id]
    assert index.total_checkpoints == 1
    assert index.latest_checkpoint_id == second.checkpoint_id
    assert len(checkpoint_files(store)) == 1
    assert await store.checkpoint_exists(first.checkpoint_id) is False


@pytest.mark.asyncio
async def test_delete_latest_repoints_latest_to_previous(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    first = make_checkpoint(data_buffer={"step": 1})
    second = make_checkpoint(data_buffer={"step": 2})
    await store.save_checkpoint(first)
    await store.save_checkpoint(second)

    await store.delete_checkpoint(second.checkpoint_id)

    index = await store.load_index()
    assert index is not None
    assert index.latest_checkpoint_id == first.checkpoint_id
    assert await store.load_checkpoint() is not None


@pytest.mark.asyncio
async def test_delete_missing_checkpoint_returns_false(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    assert await store.delete_checkpoint("cp_nope") is False


# === PRUNE ===


@pytest.mark.asyncio
async def test_prune_removes_only_old_checkpoints(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    old = make_checkpoint(
        data_buffer={"step": "old"},
        created_at=(datetime.now() - timedelta(days=30)).isoformat(),
    )
    recent = make_checkpoint(data_buffer={"step": "recent"})
    await store.save_checkpoint(old)
    await store.save_checkpoint(recent)

    deleted = await store.prune_checkpoints(max_age_days=7)

    assert deleted == 1
    assert await store.checkpoint_exists(old.checkpoint_id) is False
    assert await store.checkpoint_exists(recent.checkpoint_id) is True
    index = await store.load_index()
    assert index is not None
    assert index.total_checkpoints == 1


@pytest.mark.asyncio
async def test_prune_on_empty_store_returns_zero(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    assert await store.prune_checkpoints() == 0


# === INDEX / DISK CONSISTENCY AFTER A RE-SAVE ===


def test_resaving_an_older_checkpoint_keeps_latest_and_latest_clean_in_sync() -> None:
    """Re-saving an existing id must not desync the two "latest" accessors.

    ``get_latest_clean_checkpoint()`` returns the *last* clean entry, so a
    re-saved checkpoint has to move to the end of the list. Leaving it in place
    made ``latest_checkpoint_id`` and ``get_latest_clean_checkpoint()`` name
    different checkpoints.
    """
    index = CheckpointIndex(session_id="sess_1")
    older = make_checkpoint(data_buffer={"step": 1})
    newer = make_checkpoint(data_buffer={"step": 2})

    index.add_checkpoint(older)
    index.add_checkpoint(newer)
    index.add_checkpoint(older)  # re-save the first one

    assert index.latest_checkpoint_id == older.checkpoint_id
    assert index.get_latest_clean_checkpoint().checkpoint_id == index.latest_checkpoint_id


def test_resaving_repairs_an_index_already_duplicated_by_a_collision() -> None:
    """An index damaged before this fix must converge back to one entry per id."""
    checkpoint = make_checkpoint(data_buffer={"step": 1})
    summary = CheckpointSummary.from_checkpoint(checkpoint)
    damaged = CheckpointIndex(session_id="sess_1", checkpoints=[summary, summary.model_copy()])
    assert len(damaged.checkpoints) == 2

    damaged.add_checkpoint(checkpoint)

    ids = [s.checkpoint_id for s in damaged.checkpoints]
    assert len(ids) == len(set(ids)) == 1
    assert damaged.total_checkpoints == 1


@pytest.mark.asyncio
async def test_resaving_converges_index_onto_disk(tmp_path: Path) -> None:
    """End-to-end: a re-save leaves exactly one index entry per file on disk."""
    store = CheckpointStore(tmp_path)
    first = make_checkpoint(data_buffer={"step": 1})
    second = make_checkpoint(data_buffer={"step": 2})
    await store.save_checkpoint(first)
    await store.save_checkpoint(second)
    await store.save_checkpoint(first)  # re-save the earlier checkpoint

    index = await store.load_index()
    assert index is not None
    ids = [s.checkpoint_id for s in index.checkpoints]
    assert len(ids) == len(set(ids)) == 2
    assert index.total_checkpoints == 2 == len(checkpoint_files(store))
    assert index.latest_checkpoint_id == first.checkpoint_id
    assert (await store.load_checkpoint()).checkpoint_id == first.checkpoint_id


@pytest.mark.asyncio
async def test_legitimate_node_ids_round_trip(tmp_path: Path) -> None:
    """Node ids with dots, dashes and underscores stay usable."""
    store = CheckpointStore(tmp_path)
    checkpoint = make_checkpoint(node="my-node_01.v2")

    await store.save_checkpoint(checkpoint)
    assert await store.checkpoint_exists(checkpoint.checkpoint_id) is True
    assert await store.load_checkpoint(checkpoint.checkpoint_id) is not None
