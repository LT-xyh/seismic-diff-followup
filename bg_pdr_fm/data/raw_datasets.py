"""Raw OpenFWI and Marmousi dataset adapters retained for compatibility."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .normalization_profiles import (
    NormalizationProfile,
    get_normalization_profile,
    normalize_sparse_well_log,
    normalize_tensor,
    profile_max_min,
)
from .openfwi_lmdb import OpenFWILMDBReader, has_openfwi_lmdb_dataset, openfwi_lmdb_path, require_lmdb


DEFAULT_OPENFWI_DATASETS = ("FlatVelA", "FlatVelB", "CurveVelA", "CurveVelB", "CurveFaultA")
DEFAULT_OPENFWI_SCHEMA = ("depth_vel", "migrated_image", "horizon", "rms_vel", "well_log")
DEFAULT_MARMOUSI_SCHEMA = ("depth_vel", "migrated_image", "horizon", "rms_vel", "well_log")


class OpenFWI(Dataset):
    """Standalone OpenFWI reader used by BG-PDR-FM.

    The source dataset contains physical modalities only. `well_log` is built
    dynamically from raw `depth_vel` and is never read from disk.
    """

    def __init__(
        self,
        root_dir: str | Path = "data/openfwi",
        use_data: Iterable[str] = DEFAULT_OPENFWI_SCHEMA,
        datasets: Iterable[str] = DEFAULT_OPENFWI_DATASETS,
        use_normalize: str | None = "-1_1",
        well_count_range: tuple[int, int] | list[int] = (0, 3),
        well_seed: int = 1234,
        well_random: bool = True,
        return_well_mask: bool = True,
        storage_backend: str = "auto",
        lmdb_root: str | Path | None = None,
        normalization_profile: str = "openfwi",
        normalize_clamp: bool = True,
    ) -> None:
        self.root_dir = self._resolve_root_dir(root_dir)
        self.datasets = tuple(datasets)
        self.use_data = tuple(use_data)
        self.use_normalize = use_normalize
        self.well_count_range = self._validate_well_count_range(well_count_range)
        self.well_seed = int(well_seed)
        self.well_random = bool(well_random)
        self.return_well_mask = bool(return_well_mask)
        self.normalization_profile = str(normalization_profile)
        self.normalize_clamp = bool(normalize_clamp)
        self.normalize_max_min = profile_max_min(self.normalization_profile)
        self.file_data = self._source_modalities(self.use_data)
        self.data_files = {data_name: [] for data_name in self.file_data}
        self.lmdb_root = Path(lmdb_root) if lmdb_root is not None else self._default_lmdb_root(self.root_dir)
        self.storage_backend = self._resolve_storage_backend(storage_backend)
        self._lmdb_readers: dict[str, OpenFWILMDBReader] = {}

        if self.storage_backend == "lmdb":
            self.records = self._build_lmdb_records()
        else:
            self.records = self._build_npy_records()
        if not self.records:
            raise ValueError(f"OpenFWI dataset is empty: root={self.root_dir}, datasets={self.datasets}")

    def __len__(self) -> int:
        return len(self.records)

    def close(self) -> None:
        """Release LMDB file handles held by this dataset."""
        for reader in self._lmdb_readers.values():
            reader.close()
        self._lmdb_readers.clear()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data_dict = {}
        raw_cache = {}
        record = self.records[int(idx)]

        for data_name in self.use_data:
            if data_name == "well_log":
                depth_velocity = self._load_raw("depth_vel", record, raw_cache)
                well_log, well_mask = self._build_random_well(depth_velocity, record)
                data_dict["well_log"] = self._normalize_well_log(well_log, well_mask)
                if self.return_well_mask:
                    data_dict["well_mask"] = well_mask
                continue

            raw_data = self._load_raw(data_name, record, raw_cache)
            data = torch.from_numpy(np.asarray(raw_data)).to(torch.float32)
            data_dict[data_name] = self._normalize(data_name, data)

        return data_dict

    @staticmethod
    def _resolve_root_dir(root_dir: str | Path) -> Path:
        root = Path(root_dir)
        candidates = [root, Path.cwd() / root, Path(__file__).resolve().parents[2] / root]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        return root

    @staticmethod
    def _source_modalities(use_data: Iterable[str]) -> tuple[str, ...]:
        source_modalities = []
        for data_name in use_data:
            source_name = "depth_vel" if data_name == "well_log" else data_name
            if source_name not in source_modalities:
                source_modalities.append(source_name)
        if "depth_vel" not in source_modalities:
            source_modalities.insert(0, "depth_vel")
        return tuple(source_modalities)

    @staticmethod
    def _default_lmdb_root(root_dir: Path) -> Path:
        root = Path(root_dir)
        if root.name.lower() == "openfwi":
            return root.parent / "openfwi_lmdb"
        return root / "openfwi_lmdb"

    def _resolve_storage_backend(self, storage_backend: str) -> str:
        requested = str(storage_backend).lower()
        if requested not in {"auto", "lmdb", "npy"}:
            raise ValueError("storage_backend must be one of 'auto', 'lmdb', or 'npy'.")
        lmdb_ready = all(has_openfwi_lmdb_dataset(self.lmdb_root, name) for name in self.datasets)
        if requested == "auto":
            if lmdb_ready:
                try:
                    require_lmdb()
                except ModuleNotFoundError:
                    return "npy"
                return "lmdb"
            return "npy"
        if requested == "lmdb":
            require_lmdb()
            missing = [name for name in self.datasets if not has_openfwi_lmdb_dataset(self.lmdb_root, name)]
            if missing:
                raise FileNotFoundError(f"Missing OpenFWI LMDB datasets under {self.lmdb_root}: {missing}")
        return requested

    def _build_npy_records(self) -> list[dict]:
        records = []
        for dataset_id, dataset_name in enumerate(self.datasets):
            data_dir = self.root_dir / dataset_name
            if not data_dir.is_dir():
                raise FileNotFoundError(f"Missing OpenFWI dataset directory: {data_dir}")
            modality_files = {}
            for data_name in self.file_data:
                modality_dir = data_dir / data_name
                if not modality_dir.is_dir():
                    raise FileNotFoundError(f"Missing OpenFWI modality directory: {modality_dir}")
                files = sorted(path for path in modality_dir.iterdir() if path.suffix == ".npy")
                if not files:
                    raise FileNotFoundError(f"No .npy files found in OpenFWI modality directory: {modality_dir}")
                modality_files[data_name] = files
                self.data_files[data_name].extend(files)

            lengths = {name: len(files) for name, files in modality_files.items()}
            if len(set(lengths.values())) != 1:
                raise ValueError(f"Mismatched OpenFWI modality lengths in {data_dir}: {lengths}")

            sample_count = next(iter(lengths.values()))
            for sample_index in range(sample_count):
                records.append(
                    {
                        "dataset_name": dataset_name,
                        "dataset_id": dataset_id,
                        "sample_index": sample_index,
                        "paths": {name: files[sample_index] for name, files in modality_files.items()},
                    }
                )
        return records

    def _build_lmdb_records(self) -> list[dict]:
        records = []
        for dataset_id, dataset_name in enumerate(self.datasets):
            reader = OpenFWILMDBReader(openfwi_lmdb_path(self.lmdb_root, dataset_name))
            missing = [name for name in self.file_data if name not in reader.manifest["modalities"]]
            if missing:
                raise KeyError(f"OpenFWI LMDB {dataset_name} is missing modalities required by loader: {missing}")
            self._lmdb_readers[dataset_name] = reader
            for sample_index in range(len(reader)):
                records.append(
                    {
                        "dataset_name": dataset_name,
                        "dataset_id": dataset_id,
                        "sample_index": sample_index,
                        "paths": {},
                    }
                )
        return records

    def _load_raw(self, data_name: str, record: dict, raw_cache: dict) -> np.ndarray:
        source_name = "depth_vel" if data_name == "well_log" else data_name
        if source_name in raw_cache:
            return raw_cache[source_name]
        if self.storage_backend == "lmdb":
            raw_cache[source_name] = self._lmdb_readers[record["dataset_name"]].get(record["sample_index"], source_name)
        else:
            raw_cache[source_name] = np.load(record["paths"][source_name], allow_pickle=False)
        return raw_cache[source_name]

    def _normalize(self, data_name: str, data: torch.Tensor) -> torch.Tensor:
        return normalize_tensor(
            data,
            data_name,
            profile=self.normalization_profile,
            mode=self.use_normalize,
            clamp=self.normalize_clamp,
        )

    def _normalize_well_log(self, well_log: torch.Tensor, well_mask: torch.Tensor) -> torch.Tensor:
        return normalize_sparse_well_log(
            well_log,
            well_mask,
            profile=self.normalization_profile,
            mode=self.use_normalize,
            clamp=self.normalize_clamp,
        )

    def _well_seed(self, record: dict) -> int:
        return self.well_seed + int(record["dataset_id"]) * 1_000_000 + int(record["sample_index"])

    def _build_random_well(self, depth_velocity, record: dict) -> tuple[torch.Tensor, torch.Tensor]:
        depth = torch.from_numpy(np.asarray(depth_velocity)).to(torch.float32)
        well_log = torch.zeros_like(depth)
        well_mask = torch.zeros_like(depth)
        width = int(depth.shape[-1])
        min_count, max_count = self.well_count_range
        min_count = min(min_count, width)
        max_count = min(max_count, width)

        generator = None
        if not self.well_random:
            generator = torch.Generator().manual_seed(self._well_seed(record))
        count = int(torch.randint(min_count, max_count + 1, (1,), generator=generator).item())
        if count <= 0:
            return well_log, well_mask

        columns = torch.randperm(width, generator=generator)[:count]
        well_log[..., columns] = depth[..., columns]
        well_mask[..., columns] = 1.0
        return well_log, well_mask

    @staticmethod
    def _validate_well_count_range(well_count_range: tuple[int, int] | list[int]) -> tuple[int, int]:
        if len(well_count_range) != 2:
            raise ValueError("well_count_range must contain exactly two integers.")
        min_count, max_count = int(well_count_range[0]), int(well_count_range[1])
        if min_count < 0 or max_count < min_count:
            raise ValueError(f"Invalid well_count_range: {well_count_range!r}.")
        return min_count, max_count

    @staticmethod
    def collate_fn(batch: Iterable[dict]):
        keys = batch[0].keys()
        return {key: torch.stack([item[key] for item in batch]) for key in keys}


class Marmousi(Dataset):
    """Standalone Marmousi patch reader used by BG-PDR-FM."""

    def __init__(
        self,
        root_dir: str | Path = "data/marmousi",
        split: str = "train",
        use_data: Iterable[str] = DEFAULT_MARMOUSI_SCHEMA,
        use_normalize: str | None = "-1_1",
        return_metadata: bool = False,
        metadata_keys: Iterable[str] = (
            "positions",
            "patch_x_ranges_m",
            "patch_z_ranges_m",
            "time_windows_ms",
            "patch_twt_ranges_ms",
        ),
        well_count_range: tuple[int, int] | list[int] = (0, 3),
        well_seed: int = 1234,
        well_random: bool | None = None,
        return_well_mask: bool = True,
        validate_files: bool = True,
        normalize_max_min: dict[str, tuple[float, float]] | None = None,
        normalization_profile: str = "marmousi",
        normalize_clamp: bool = True,
    ) -> None:
        self.root_dir = self._resolve_root_dir(root_dir)
        self.meta_path = self.root_dir / "dataset_meta.json"
        self.meta = self._load_meta(self.meta_path)

        if split not in self.meta.get("splits", {}):
            available_splits = sorted(self.meta.get("splits", {}).keys())
            raise ValueError(f"Unknown split {split!r}. Available splits: {available_splits}")

        self.split = split
        self.split_dir = self.root_dir / split
        self.use_data = tuple(use_data)
        self.use_normalize = use_normalize
        self.return_metadata = return_metadata
        self.metadata_keys = tuple(metadata_keys)
        self.well_count_range = OpenFWI._validate_well_count_range(well_count_range)
        self.well_seed = int(well_seed)
        self.well_random = self.split == "train" if well_random is None else bool(well_random)
        self.return_well_mask = bool(return_well_mask)
        self.normalize_clamp = bool(normalize_clamp)
        profile = get_normalization_profile(normalization_profile)
        profile_ranges = dict(profile.ranges)
        if normalize_max_min is not None:
            for key, (max_value, min_value) in normalize_max_min.items():
                profile_ranges[key] = (float(min_value), float(max_value))
        self.normalization_profile = NormalizationProfile(
            name=profile.name,
            ranges=profile_ranges,
            discrete_modalities=profile.discrete_modalities,
        )
        self.normalize_max_min = profile_max_min(self.normalization_profile)

        self.available_modalities = tuple(self.meta["layout"]["modalities"].keys())
        self.data_files = self._build_data_files(validate_files=validate_files)
        self.metadata = self._load_split_metadata() if return_metadata else {}

    def __len__(self) -> int:
        if not self.use_data:
            return int(self.meta["splits"][self.split]["count"])
        return len(self.data_files[self.use_data[0]])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data_dict = {}
        raw_cache = {}

        for data_name in self.use_data:
            if data_name == "well_log":
                depth_velocity = self._load_raw("well_log", idx, raw_cache)
                well_log, well_mask = self._build_random_well(depth_velocity, idx)
                data_dict["well_log"] = self._normalize_well_log(well_log, well_mask)
                if self.return_well_mask:
                    data_dict["well_mask"] = well_mask
                continue

            data = self._load_raw(data_name, idx, raw_cache)
            tensor = torch.from_numpy(np.asarray(data)).to(torch.float32)
            data_dict[data_name] = self._normalize(data_name, tensor)

        if self.return_metadata:
            for key, values in self.metadata.items():
                data_dict[key] = torch.from_numpy(np.asarray(values[idx])).to(torch.float32)

        return data_dict

    def get_sample_paths(self, idx: int) -> dict[str, Path]:
        return {data_name: self.data_files[data_name][idx] for data_name in self.use_data}

    def _build_data_files(self, validate_files: bool) -> dict[str, list[Path]]:
        data_files = {}
        for data_name in self.use_data:
            source_name = self._resolve_source_modality(data_name)
            files = self._modality_files(source_name, validate_files=validate_files)
            data_files[data_name] = files

        lengths = {data_name: len(files) for data_name, files in data_files.items()}
        if lengths and len(set(lengths.values())) != 1:
            raise ValueError(f"Mismatched Marmousi modality lengths: {lengths}")

        return data_files

    def _resolve_source_modality(self, data_name: str) -> str:
        if data_name == "well_log":
            return "depth_vel"
        if data_name in self.available_modalities:
            return data_name
        raise ValueError(
            f"Modality {data_name!r} is not available in {self.root_dir}. "
            f"Available modalities: {self.available_modalities}"
        )

    def _modality_files(self, data_name: str, validate_files: bool) -> list[Path]:
        split_meta = self.meta["splits"][self.split]
        sample_index_start = int(self.meta.get("layout", {}).get("sample_index_start", 0))
        count = int(split_meta["count"])
        modality_dir = self.split_dir / data_name

        files = [modality_dir / f"{idx}.npy" for idx in range(sample_index_start, sample_index_start + count)]
        if validate_files:
            missing = [str(path) for path in files if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing {len(missing)} files for {self.split}/{data_name}: {missing[:5]}")
        return files

    def _load_split_metadata(self) -> dict[str, np.ndarray]:
        split_meta = self.meta["splits"][self.split]
        metadata_files = split_meta.get("metadata_files", {})
        fallback_files = {
            "positions": "positions_iz_ix.npy",
            "patch_x_ranges_m": "patch_x_ranges_m.npy",
            "patch_z_ranges_m": "patch_z_ranges_m.npy",
            "time_windows_ms": "time_windows_ms.npy",
            "patch_twt_ranges_ms": "patch_twt_ranges_ms.npy",
        }

        metadata = {}
        for key in self.metadata_keys:
            raw_path = metadata_files.get(key)
            path = self._resolve_meta_file(raw_path) if raw_path else self.split_dir / fallback_files[key]
            values = np.load(path)
            if len(values) != len(self):
                raise ValueError(f"Metadata {key!r} length {len(values)} does not match dataset length {len(self)}.")
            metadata[key] = values
        return metadata

    def _resolve_meta_file(self, raw_path) -> Path:
        path = Path(raw_path)
        if path.is_file():
            return path
        candidate = self.split_dir / path.name
        if candidate.is_file():
            return candidate
        return path

    def _load_raw(self, data_name: str, idx: int, raw_cache: dict) -> np.ndarray:
        source_name = self._resolve_source_modality(data_name)
        if source_name not in raw_cache:
            raw_cache[source_name] = np.load(self.data_files[data_name][idx], allow_pickle=False)
        return raw_cache[source_name]

    def _build_random_well(self, depth_velocity, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        depth = torch.from_numpy(np.asarray(depth_velocity)).to(torch.float32)
        well_log = torch.zeros_like(depth)
        well_mask = torch.zeros_like(depth)
        width = int(depth.shape[-1])
        min_count, max_count = self.well_count_range
        min_count = min(min_count, width)
        max_count = min(max_count, width)

        generator = None
        if not self.well_random:
            generator = torch.Generator().manual_seed(self.well_seed + int(idx))
        count = int(torch.randint(min_count, max_count + 1, (1,), generator=generator).item())
        if count <= 0:
            return well_log, well_mask

        columns = torch.randperm(width, generator=generator)[:count]
        well_log[..., columns] = depth[..., columns]
        well_mask[..., columns] = 1.0
        return well_log, well_mask

    def _normalize_well_log(self, well_log: torch.Tensor, well_mask: torch.Tensor) -> torch.Tensor:
        return normalize_sparse_well_log(
            well_log,
            well_mask,
            profile=self.normalization_profile,
            mode=self.use_normalize,
            clamp=self.normalize_clamp,
        )

    def _normalize(self, data_name: str, data: torch.Tensor) -> torch.Tensor:
        return normalize_tensor(
            data,
            data_name,
            profile=self.normalization_profile,
            mode=self.use_normalize,
            clamp=self.normalize_clamp,
        )

    @staticmethod
    def _resolve_root_dir(root_dir: str | Path) -> Path:
        root = Path(root_dir)
        candidates = [root, Path.cwd() / root, Path(__file__).resolve().parents[2] / root]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError(f"Cannot find Marmousi root directory: {root_dir}")

    @staticmethod
    def _load_meta(meta_path: Path) -> dict:
        if not meta_path.is_file():
            raise FileNotFoundError(f"Cannot find Marmousi metadata file: {meta_path}")
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def collate_fn(batch: Iterable[dict]):
        keys = batch[0].keys()
        return {key: torch.stack([item[key] for item in batch]) for key in keys}
