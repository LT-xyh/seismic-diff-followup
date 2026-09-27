"""OpenFWI LMDB reader and writer helpers."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_OPENFWI_LMDB_MODALITIES = ("depth_vel", "migrated_image", "horizon", "rms_vel")
MANIFEST_NAME = "manifest.json"
MANIFEST_KEY = b"__manifest__"


def require_lmdb():
    try:
        import lmdb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenFWI LMDB backend requires the `lmdb` package. "
            "Install `lmdb` or run with storage_backend='npy'."
        ) from exc
    return lmdb


def openfwi_lmdb_path(lmdb_root: str | Path, dataset_name: str) -> Path:
    return Path(lmdb_root) / f"{dataset_name}.lmdb"


def has_openfwi_lmdb_dataset(lmdb_root: str | Path, dataset_name: str) -> bool:
    path = openfwi_lmdb_path(lmdb_root, dataset_name)
    return path.is_dir() and (path / MANIFEST_NAME).is_file()


def sample_key(sample_index: int, modality: str) -> bytes:
    return f"{int(sample_index):08d}:{modality}".encode("ascii")


def load_manifest(lmdb_path: str | Path) -> dict:
    manifest_path = Path(lmdb_path) / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"OpenFWI LMDB manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


class OpenFWILMDBReader:
    _ENV_CACHE = {}
    _ENV_REF_COUNTS = {}

    def __init__(self, lmdb_path: str | Path) -> None:
        self.path = Path(lmdb_path)
        self.manifest = load_manifest(self.path)
        self._env = None
        self._cache_key = str(self.path.resolve())

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_env"] = None
        return state

    def close(self) -> None:
        if self._env is not None:
            count = int(self._ENV_REF_COUNTS.get(self._cache_key, 0)) - 1
            if count <= 0:
                self._env.close()
                self._ENV_CACHE.pop(self._cache_key, None)
                self._ENV_REF_COUNTS.pop(self._cache_key, None)
            else:
                self._ENV_REF_COUNTS[self._cache_key] = count
        self._env = None

    def _open(self):
        if self._env is None:
            env = self._ENV_CACHE.get(self._cache_key)
            if env is None:
                lmdb = require_lmdb()
                env = lmdb.open(
                    str(self.path),
                    readonly=True,
                    lock=False,
                    readahead=True,
                    meminit=False,
                    max_readers=2048,
                )
                self._ENV_CACHE[self._cache_key] = env
                self._ENV_REF_COUNTS[self._cache_key] = 0
            self._ENV_REF_COUNTS[self._cache_key] = int(self._ENV_REF_COUNTS[self._cache_key]) + 1
            self._env = env
        return self._env

    def __len__(self) -> int:
        return int(self.manifest["sample_count"])

    def get(self, sample_index: int, modality: str) -> np.ndarray:
        if modality not in self.manifest["modalities"]:
            raise KeyError(f"OpenFWI LMDB {self.path} does not contain modality {modality!r}.")
        env = self._open()
        with env.begin(write=False) as txn:
            value = txn.get(sample_key(sample_index, modality))
        if value is None:
            raise KeyError(f"Missing OpenFWI LMDB key {sample_key(sample_index, modality)!r} in {self.path}.")
        array = np.load(io.BytesIO(value), allow_pickle=False)
        return np.asarray(array)


def _npy_files(modality_dir: Path) -> list[Path]:
    if not modality_dir.is_dir():
        raise FileNotFoundError(f"Missing modality directory: {modality_dir}")
    files = sorted(path for path in modality_dir.iterdir() if path.suffix == ".npy")
    if not files:
        raise FileNotFoundError(f"No .npy files found in modality directory: {modality_dir}")
    return files


def _estimate_map_size(file_lists: dict[str, list[Path]], safety_factor: float = 1.35) -> int:
    total = sum(path.stat().st_size for files in file_lists.values() for path in files)
    return max(int(total * safety_factor), 1 << 30)


def _read_npy_metadata(path: Path) -> dict[str, object]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def build_manifest(
    *,
    dataset_name: str,
    source_dir: str | Path,
    sample_count: int,
    modalities: Iterable[str],
    normalization_profile: str = "openfwi",
) -> dict:
    source_dir = Path(source_dir)
    modality_meta = {
        modality: _read_npy_metadata(_npy_files(source_dir / modality)[0])
        for modality in modalities
    }
    return {
        "format": "openfwi_lmdb",
        "version": 1,
        "dataset_name": dataset_name,
        "sample_count": int(sample_count),
        "modalities": list(modalities),
        "modality_meta": modality_meta,
        "source_root": str(source_dir.resolve()),
        "normalization_profile": normalization_profile,
        "key_format": "{sample_index:08d}:{modality}",
        "stores_well_log": False,
    }


def write_openfwi_lmdb_dataset(
    *,
    source_dataset_dir: str | Path,
    lmdb_path: str | Path,
    dataset_name: str | None = None,
    modalities: Iterable[str] = DEFAULT_OPENFWI_LMDB_MODALITIES,
    normalization_profile: str = "openfwi",
    map_size: int | None = None,
    overwrite: bool = False,
    commit_interval: int = 512,
) -> dict:
    lmdb = require_lmdb()
    source_dataset_dir = Path(source_dataset_dir)
    lmdb_path = Path(lmdb_path)
    dataset_name = dataset_name or source_dataset_dir.name
    modalities = tuple(modalities)

    file_lists = {modality: _npy_files(source_dataset_dir / modality) for modality in modalities}
    lengths = {modality: len(files) for modality, files in file_lists.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Mismatched OpenFWI modality lengths for {source_dataset_dir}: {lengths}")
    sample_count = next(iter(lengths.values()))

    if lmdb_path.is_file():
        raise FileExistsError(f"LMDB path exists as a file, expected a directory: {lmdb_path}")
    if lmdb_path.exists() and any(lmdb_path.iterdir()) and not overwrite:
        raise FileExistsError(f"LMDB path already exists and is not empty: {lmdb_path}")
    lmdb_path.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        dataset_name=dataset_name,
        source_dir=source_dataset_dir,
        sample_count=sample_count,
        modalities=modalities,
        normalization_profile=normalization_profile,
    )
    map_size = int(map_size or _estimate_map_size(file_lists))

    commit_interval = max(1, int(commit_interval))
    env = lmdb.open(str(lmdb_path), map_size=map_size, subdir=True, lock=True, readahead=False, meminit=False)
    try:
        with env.begin(write=True) as txn:
            txn.put(MANIFEST_KEY, json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
        for start in range(0, sample_count, commit_interval):
            stop = min(start + commit_interval, sample_count)
            with env.begin(write=True) as txn:
                for sample_index in range(start, stop):
                    for modality, files in file_lists.items():
                        source_file = files[sample_index]
                        if not source_file.is_file():
                            raise FileNotFoundError(f"Source file disappeared while writing LMDB: {source_file}")
                        txn.put(sample_key(sample_index, modality), source_file.read_bytes())
        env.sync()
    finally:
        env.close()

    (lmdb_path / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def prepare_openfwi_lmdb_root(
    *,
    source_root: str | Path,
    lmdb_root: str | Path,
    dataset_names: Iterable[str],
    split_seed: int,
    well_seed: int,
    modalities: Iterable[str] = DEFAULT_OPENFWI_LMDB_MODALITIES,
    normalization_profile: str = "openfwi",
    overwrite: bool = False,
) -> dict:
    """Build or verify LMDB subsets and return reproducibility provenance."""
    source_root = Path(source_root)
    lmdb_root = Path(lmdb_root)
    modalities = tuple(str(modality) for modality in modalities)
    datasets: dict[str, dict[str, object]] = {}
    for dataset_name in dataset_names:
        name = str(dataset_name)
        source_dataset_dir = source_root / name
        file_lists = {modality: _npy_files(source_dataset_dir / modality) for modality in modalities}
        lengths = {modality: len(files) for modality, files in file_lists.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Mismatched OpenFWI modality lengths for {source_dataset_dir}: {lengths}")
        lmdb_path = openfwi_lmdb_path(lmdb_root, name)
        if has_openfwi_lmdb_dataset(lmdb_root, name) and not overwrite:
            manifest = load_manifest(lmdb_path)
            if int(manifest["sample_count"]) != next(iter(lengths.values())):
                raise ValueError(
                    f"Existing LMDB sample count differs from source for {name}: "
                    f"{manifest['sample_count']} != {next(iter(lengths.values()))}. Use --force to rebuild."
                )
        else:
            manifest = write_openfwi_lmdb_dataset(
                source_dataset_dir=source_dataset_dir,
                lmdb_path=lmdb_path,
                dataset_name=name,
                modalities=modalities,
                normalization_profile=normalization_profile,
                overwrite=overwrite,
            )
        datasets[name] = {
            "sample_count": int(manifest["sample_count"]),
            "modalities": list(manifest["modalities"]),
            "input_sha256": _input_sha256(file_lists),
            "lmdb_path": str(lmdb_path),
        }
    return {
        "version": 1,
        "source_root": str(source_root),
        "lmdb_root": str(lmdb_root),
        "split_seed": int(split_seed),
        "well_seed": int(well_seed),
        "datasets": datasets,
    }


def _input_sha256(file_lists: dict[str, list[Path]]) -> str:
    digest = hashlib.sha256()
    for modality in sorted(file_lists):
        for path in file_lists[modality]:
            digest.update(f"{modality}/{path.name}\n".encode("utf-8"))
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
    return digest.hexdigest()
