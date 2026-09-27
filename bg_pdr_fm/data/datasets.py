"""Dataset adapters for BG-PDR-FM.

This module turns the raw dataset layouts used in this repo into a fixed sample
schema that the BG-PDR-FM models can consume directly.

Typical usage:

    from bg_pdr_fm.data.datasets import OpenFWIBGDataset
    from torch.utils.data import DataLoader

    dataset = OpenFWIBGDataset(
        root_dir="data/openfwi",
        datasets=("FlatVelA",),
        split="train",
        split_fractions=(0.7, 0.2, 0.1),
        split_seed=42,
        storage_backend="auto",
        lmdb_root="I:/Datasets/openfwi_lmdb",
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

Each sample always contains:
    depth_vel, migrated_image, horizon, rms_vel, well_log, well_mask,
    modality_mask, modality_quality

Important conventions:
    - `well_log` is not stored on disk. It is generated dynamically from raw
      `depth_vel` by sampling 0-3 well columns.
    - `well_mask` marks the sampled columns explicitly and is required by the
      model and metrics.
    - OpenFWI can read from either `.npy` files or per-subset LMDB stores.
    - Marmousi keeps the file-per-sample layout and preserves the physical
      shape returned by the standalone `bg_pdr_fm.data.Marmousi` reader. In particular,
      time-domain modalities such as `migrated_image` and `rms_vel` may be
      `(1, 1000, 70)` while depth-domain fields remain `(1, 70, 70)`.
    - Train/val/test membership is shuffled once with `split_seed`, then kept
      deterministic for reproducible experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .batch import STANDARD_MODALITIES
except ImportError:
    from batch import STANDARD_MODALITIES

from .normalization_profiles import normalize_sparse_well_log as _profile_normalize_sparse_well_log
from .normalization_profiles import normalize_tensor as _profile_normalize_tensor
from .normalization_profiles import profile_max_min
from .openfwi_lmdb import OpenFWILMDBReader, has_openfwi_lmdb_dataset, openfwi_lmdb_path, require_lmdb
from .raw_datasets import Marmousi


DEFAULT_OPENFWI_DATASETS = ("FlatVelA", "FlatVelB", "CurveVelA", "CurveVelB", "CurveFaultA")
OPENFWI_NORMALIZE_MAX_MIN = profile_max_min("openfwi")


def _as_path_list(paths: str | Path | list[str] | tuple[str, ...] | None) -> list[Path]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(path) for path in paths]


def _standardize_tensor(
    key: str,
    value: torch.Tensor,
    target_shape: tuple[int, int] | None,
    context: str,
) -> torch.Tensor:
    """Convert raw arrays to CHW float tensors without changing physical shape.

    `target_shape` is kept only as an explicit compatibility escape hatch. The default
    is `None`, so dataset adapters preserve the shapes loaded by
    the standalone BG-PDR-FM dataset readers.
    """
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3:
        raise ValueError(f"{context} key {key!r} must be CHW or HW, got {tuple(value.shape)}.")
    value = value.to(torch.float32)
    if target_shape is None or tuple(value.shape[-2:]) == target_shape:
        return value.contiguous()

    import torch.nn.functional as F

    mode = "nearest" if key in {"horizon", "well_mask"} else "bilinear"
    kwargs = {} if mode == "nearest" else {"align_corners": False}
    resized = F.interpolate(value.unsqueeze(0), size=target_shape, mode=mode, **kwargs).squeeze(0)
    return resized.contiguous()


def _validate_well_count_range(well_count_range: tuple[int, int] | list[int]) -> tuple[int, int]:
    if len(well_count_range) != 2:
        raise ValueError("well_count_range must contain exactly two integers.")
    min_count, max_count = int(well_count_range[0]), int(well_count_range[1])
    if min_count < 0 or max_count < min_count:
        raise ValueError(f"Invalid well_count_range: {well_count_range!r}.")
    return min_count, max_count


def _validate_split_fractions(split_fractions: tuple[float, float, float] | list[float]) -> tuple[float, float, float]:
    """Validate train/val/test fractions and normalize minor round-off drift."""
    if len(split_fractions) != 3:
        raise ValueError("split_fractions must contain exactly three values: train, val, test.")
    train_fraction, val_fraction, test_fraction = (float(v) for v in split_fractions)
    if min(train_fraction, val_fraction, test_fraction) < 0:
        raise ValueError(f"split_fractions cannot contain negative values: {split_fractions!r}.")
    total = train_fraction + val_fraction + test_fraction
    if total <= 0:
        raise ValueError(f"split_fractions must sum to a positive value: {split_fractions!r}.")
    return train_fraction / total, val_fraction / total, test_fraction / total


def _shuffled_indices(total: int, seed: int) -> list[int]:
    """Return a deterministic random permutation of sample indices."""
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randperm(int(total), generator=generator).tolist()


def _build_random_well(
    depth_velocity: torch.Tensor,
    idx: int,
    well_count_range: tuple[int, int],
    well_seed: int,
    well_random: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a sparse well constraint from a raw depth velocity field.

    The observation is column-wise: a chosen column copies the raw velocity
    values into `well_log`, and the same column is marked as valid in
    `well_mask`. Unobserved locations stay exactly zero.
    """
    depth = depth_velocity.to(torch.float32)
    well_log = torch.zeros_like(depth)
    well_mask = torch.zeros_like(depth)
    width = int(depth.shape[-1])
    min_count, max_count = well_count_range
    min_count = min(min_count, width)
    max_count = min(max_count, width)

    generator = None
    if not well_random:
        generator = torch.Generator().manual_seed(int(well_seed) + int(idx))
    count = int(torch.randint(min_count, max_count + 1, (1,), generator=generator).item())
    if count <= 0:
        return well_log, well_mask

    columns = torch.randperm(width, generator=generator)[:count]
    well_log[..., columns] = depth[..., columns]
    well_mask[..., columns] = 1.0
    return well_log.contiguous(), well_mask.contiguous()


def _normalize_sparse_well_log(
    well_log: torch.Tensor,
    well_mask: torch.Tensor,
    use_normalize: str | None,
    normalize_max_min: dict[str, tuple[float, float]],
    normalization_profile: str = "openfwi",
    normalize_clamp: bool = True,
) -> torch.Tensor:
    """Normalize only the observed well pixels and keep the rest at zero."""
    del normalize_max_min
    return _profile_normalize_sparse_well_log(
        well_log,
        well_mask,
        profile=normalization_profile,
        mode=use_normalize,
        clamp=normalize_clamp,
    )


def _has_well_observation(well_mask: torch.Tensor) -> float:
    return float(bool((well_mask > 0).any().item()))


def resolve_marmousi_root(
    root_dir: str | Path | None = None,
    root_candidates: list[str] | tuple[str, ...] | None = None,
    include_defaults: bool = True,
) -> Path:
    """Find the Marmousi root that contains `dataset_meta.json`.

    The adapter accepts a user-provided path, candidate paths from config, and a
    couple of repo-local defaults. The first directory with the metadata file is
    used.
    """
    repo_root = Path(__file__).resolve().parents[2]
    candidates: list[Path] = []
    candidates.extend(_as_path_list(root_dir))
    candidates.extend(_as_path_list(root_candidates))
    if include_defaults:
        candidates.append(repo_root / "data" / "marmousi")

    checked: list[str] = []
    for raw in candidates:
        path = raw if raw.is_absolute() else repo_root / raw
        path = path.resolve()
        checked.append(str(path))
        if (path / "dataset_meta.json").is_file():
            return path

    joined = "\n  - ".join(checked)
    raise FileNotFoundError(
        "Cannot find Marmousi dataset root for BG-PDR-FM. Expected a directory containing "
        f"`dataset_meta.json`. Checked:\n  - {joined}\n"
        "Set `data.root_dir` or add an absolute path under `data.root_candidates` in "
        "`bg_pdr_fm/configs/bg_pdr_fm.yaml`."
    )


def resolve_openfwi_root(
    root_dir: str | Path | None = None,
    root_candidates: list[str] | tuple[str, ...] | None = None,
    datasets: list[str] | tuple[str, ...] | None = None,
    required_modalities: list[str] | tuple[str, ...] | None = None,
    include_defaults: bool = True,
) -> Path:
    """Find the OpenFWI source root that contains the expected subsets.

    The returned directory should contain one subdirectory per OpenFWI subset,
    such as `FlatVelA/` or `CurveFaultA/`.
    """
    repo_root = Path(__file__).resolve().parents[2]
    expected_datasets = tuple(datasets or DEFAULT_OPENFWI_DATASETS)
    required = tuple(modality for modality in (required_modalities or ("depth_vel",)) if modality != "well_log")
    if "depth_vel" not in required:
        required = ("depth_vel", *required)
    candidates: list[Path] = []
    candidates.extend(_as_path_list(root_dir))
    candidates.extend(_as_path_list(root_candidates))
    if include_defaults:
        candidates.append(repo_root / "data" / "openfwi")

    checked: list[str] = []
    reports: list[str] = []
    for raw in candidates:
        path = raw if raw.is_absolute() else repo_root / raw
        path = path.resolve()
        checked.append(str(path))
        missing: list[str] = []
        for dataset_name in expected_datasets:
            dataset_dir = path / dataset_name
            if not dataset_dir.is_dir():
                missing.append(f"{dataset_name}/")
                continue
            for modality in required:
                modality_dir = dataset_dir / modality
                has_files = modality_dir.is_dir() and any(modality_dir.glob("*.npy"))
                if not has_files:
                    missing.append(f"{dataset_name}/{modality}/*.npy")
        if not missing:
            return path
        reports.append(f"{path}: missing {', '.join(missing[:12])}")

    joined = "\n  - ".join(checked)
    report = "\n  - ".join(reports)
    raise FileNotFoundError(
        "Cannot find OpenFWI dataset root for BG-PDR-FM. Expected sub-datasets "
        f"{list(expected_datasets)} with required modality folders {list(required)}. Checked:\n  - {joined}\n"
        f"Missing details:\n  - {report}\n"
        "Set `data.root_dir` or add an absolute path under `data.root_candidates` in "
        "`bg_pdr_fm/configs/bg_pdr_fm.yaml`."
    )


def default_openfwi_lmdb_root(openfwi_root: str | Path) -> Path:
    """Default LMDB root that lives next to the OpenFWI source tree."""
    root = Path(openfwi_root)
    if root.name.lower() == "openfwi":
        return root.parent / "openfwi_lmdb"
    return root / "openfwi_lmdb"


class MarmousiBGDataset(Dataset):
    """Adapter that converts Marmousi samples into BG-PDR-FM items.

    Use this when you want Marmousi samples to follow the same batch schema as
    OpenFWI without rewriting the original Marmousi loader.

    Marmousi already has physical `train/` and `test/` folders on disk. To avoid
    mixing the independent test grid back into training, this adapter keeps
    `test` mapped to the physical test folder and derives `train`/`val` from the
    physical train folder.
    """

    def __init__(
        self,
        root_dir: str = "data/marmousi",
        root_candidates: list[str] | tuple[str, ...] | None = None,
        split: str = "train",
        use_normalize: str | None = "-1_1",
        target_shape: tuple[int, int] | list[int] | None = None,
        return_metadata: bool = False,
        split_fractions: tuple[float, float, float] | list[float] = (0.7, 0.2, 0.1),
        split_seed: int = 42,
        well_count_range: tuple[int, int] | list[int] = (0, 3),
        well_seed: int = 1234,
        well_random: bool | None = None,
        normalization_profile: str = "marmousi",
        normalize_clamp: bool = True,
        **dataset_kwargs: Any,
    ) -> None:
        resolved_root = resolve_marmousi_root(root_dir=root_dir, root_candidates=root_candidates)
        self.root_dir = resolved_root
        self.split = str(split)
        self.source_split = "train" if self.split in {"train", "val", "valid"} else self.split
        self.split_fractions = _validate_split_fractions(split_fractions)
        self.split_seed = int(split_seed)
        self.well_count_range = _validate_well_count_range(well_count_range)
        self.well_seed = int(well_seed)
        self.well_random = self.split == "train" if well_random is None else bool(well_random)
        self.target_shape = tuple(int(v) for v in target_shape) if target_shape is not None else None
        self.dataset = Marmousi(
            root_dir=str(resolved_root),
            split=self.source_split,
            use_data=("depth_vel", "migrated_image", "horizon", "rms_vel", "well_log"),
            use_normalize=use_normalize,
            return_metadata=return_metadata,
            well_count_range=self.well_count_range,
            well_seed=self.well_seed,
            well_random=self.well_random,
            return_well_mask=True,
            normalization_profile=normalization_profile,
            normalize_clamp=normalize_clamp,
            **dataset_kwargs,
        )
        if len(self.dataset) <= 0:
            raise ValueError(f"MarmousiBGDataset at {resolved_root} split={split!r} is empty.")
        self.indices = self._select_indices(len(self.dataset), self.split, self.split_fractions, self.split_seed)
        if not self.indices:
            raise ValueError(f"MarmousiBGDataset at {resolved_root} split={split!r} is empty after virtual split.")

    def __len__(self) -> int:
        return len(self.indices)

    @staticmethod
    def _select_indices(
        total: int,
        split: str,
        split_fractions: tuple[float, float, float],
        split_seed: int,
    ) -> list[int]:
        """Select random but reproducible indices without mixing physical folders."""
        indices = _shuffled_indices(total, split_seed)
        if split == "all":
            return indices
        if split == "test":
            return indices
        train_fraction, val_fraction, _ = split_fractions
        train_val_total = train_fraction + val_fraction
        if train_val_total <= 0:
            raise ValueError(f"Invalid train/val split fractions: {split_fractions!r}.")
        train_ratio_inside_physical_train = train_fraction / train_val_total
        train_end = int(round(total * train_ratio_inside_physical_train))
        train_end = max(1, min(train_end, total))
        if train_end >= total and total >= 2:
            train_end = total - 1
        if split == "train":
            return indices[:train_end]
        if split in {"val", "valid"}:
            return indices[train_end:]
        raise ValueError("MarmousiBGDataset split must be one of 'train', 'val', 'valid', 'test', or 'all'.")

    def _standardize_tensor(self, key: str, value: torch.Tensor) -> torch.Tensor:
        return _standardize_tensor(key, value, self.target_shape, context="MarmousiBGDataset")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Load the standalone sample. If `target_shape` is explicitly set, this
        # compatibility path resizes tensors here; the default preserves native
        # physical shapes.
        source_index = int(self.indices[int(idx)])
        item = dict(self.dataset[source_index])
        for key in ("depth_vel", *STANDARD_MODALITIES, "well_mask"):
            item[key] = self._standardize_tensor(key, item[key])
        item["well_log"] = item["well_log"] * item["well_mask"]
        mask = torch.ones(len(STANDARD_MODALITIES), dtype=torch.float32)
        quality = torch.ones(len(STANDARD_MODALITIES), dtype=torch.float32)
        well_index = STANDARD_MODALITIES.index("well_log")
        mask[well_index] = _has_well_observation(item["well_mask"])
        quality[well_index] = mask[well_index]
        item["modality_mask"] = mask
        item["modality_quality"] = quality
        item["dataset_id"] = torch.tensor(0.0, dtype=torch.float32)
        item["sample_index"] = torch.tensor(float(source_index), dtype=torch.float32)
        return item


class OpenFWIBGDataset(Dataset):
    """Adapter that exposes OpenFWI samples through the BG-PDR-FM batch schema.

    Reading path:
        1. Resolve the OpenFWI source tree.
        2. Prefer LMDB if available and requested via `storage_backend`.
        3. Load `depth_vel` once.
        4. Build a sparse `well_log` and `well_mask` from that same raw tensor.
        5. Normalize all modalities with the dataset-aware profile.
    """

    def __init__(
        self,
        root_dir: str | None = "data/openfwi",
        root_candidates: list[str] | tuple[str, ...] | None = None,
        datasets: list[str] | tuple[str, ...] | None = None,
        post_split_datasets: list[str] | tuple[str, ...] | None = None,
        split: str = "train",
        use_normalize: str | None = "-1_1",
        target_shape: tuple[int, int] | list[int] | None = None,
        required_modalities: list[str] | tuple[str, ...] | None = None,
        split_fractions: tuple[float, float, float] | list[float] = (0.7, 0.2, 0.1),
        split_seed: int = 42,
        val_fraction: float | None = None,
        well_count_range: tuple[int, int] | list[int] = (0, 3),
        well_seed: int = 1234,
        well_random: bool | None = None,
        storage_backend: str = "auto",
        lmdb_root: str | Path | None = None,
        normalization_profile: str = "openfwi",
        normalize_clamp: bool = True,
    ) -> None:
        self.dataset_names = tuple(datasets or DEFAULT_OPENFWI_DATASETS)
        self.post_split_datasets = tuple(post_split_datasets) if post_split_datasets is not None else None
        if self.post_split_datasets is not None:
            unknown = sorted(set(self.post_split_datasets).difference(self.dataset_names))
            if unknown:
                raise ValueError(
                    "post_split_datasets must be a subset of datasets; "
                    f"unknown datasets: {unknown}"
                )
        self.required_modalities = tuple(
            modality for modality in (required_modalities or ("depth_vel",)) if modality != "well_log"
        )
        if "depth_vel" not in self.required_modalities:
            self.required_modalities = ("depth_vel", *self.required_modalities)
        unresolved_root = Path(root_dir) if root_dir is not None else Path("data/openfwi")
        self.lmdb_root = Path(lmdb_root) if lmdb_root is not None else default_openfwi_lmdb_root(unresolved_root)
        self.storage_backend = self._resolve_storage_backend(storage_backend)
        if self.storage_backend == "lmdb":
            # LMDB is the complete OpenFWI training source. The original `.npy`
            # tree may be archived on another machine, so source-root probing is
            # best effort instead of a hard requirement in this mode.
            self.root_dir = self._best_effort_root_dir(root_dir)
        else:
            self.root_dir = resolve_openfwi_root(
                root_dir=root_dir,
                root_candidates=root_candidates,
                datasets=self.dataset_names,
                required_modalities=self.required_modalities,
            )
        self.split = str(split)
        self.use_normalize = use_normalize
        self.normalization_profile = str(normalization_profile)
        self.normalize_clamp = bool(normalize_clamp)
        self.target_shape = tuple(int(v) for v in target_shape) if target_shape is not None else None
        self.split_fractions = _validate_split_fractions(split_fractions)
        self.split_seed = int(split_seed)
        self.well_count_range = _validate_well_count_range(well_count_range)
        self.well_seed = int(well_seed)
        self.well_random = self.split == "train" if well_random is None else bool(well_random)
        self._lmdb_readers: dict[str, OpenFWILMDBReader] = {}
        if self.storage_backend == "lmdb":
            records = self._build_lmdb_records()
        else:
            records = self._build_records()
        if val_fraction is not None:
            self.split_fractions = self._legacy_val_fraction_to_split_fractions(float(val_fraction))
        self.records = self._select_split_then_filter(
            records,
            self.split,
            self.split_fractions,
            self.split_seed,
            self.post_split_datasets,
        )
        if not self.records:
            raise ValueError(f"OpenFWIBGDataset at {self.root_dir} split={split!r} is empty.")

    @staticmethod
    def _best_effort_root_dir(root_dir: str | Path | None) -> Path:
        repo_root = Path(__file__).resolve().parents[2]
        if root_dir is None:
            return repo_root / "data" / "openfwi"
        path = Path(root_dir)
        return path if path.is_absolute() else repo_root / path

    def _resolve_storage_backend(self, storage_backend: str) -> str:
        """Choose the actual backend.

        `auto` means "use LMDB if every requested subset has a matching LMDB
        store, otherwise fall back to `.npy`".
        """
        requested = str(storage_backend).lower()
        if requested not in {"auto", "lmdb", "npy"}:
            raise ValueError("OpenFWIBGDataset storage_backend must be one of 'auto', 'lmdb', or 'npy'.")
        lmdb_ready = all(has_openfwi_lmdb_dataset(self.lmdb_root, name) for name in self.dataset_names)
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
            missing = [name for name in self.dataset_names if not has_openfwi_lmdb_dataset(self.lmdb_root, name)]
            if missing:
                raise FileNotFoundError(f"Missing OpenFWI LMDB datasets under {self.lmdb_root}: {missing}")
        return requested

    def _modality_files(self, dataset_dir: Path, modality: str) -> list[Path] | None:
        modality_dir = dataset_dir / modality
        if not modality_dir.is_dir():
            return None
        files = sorted(path for path in modality_dir.iterdir() if path.suffix == ".npy")
        return files or None

    def _build_records(self) -> list[dict[str, Any]]:
        """Build an index over the on-disk `.npy` layout."""
        records: list[dict[str, Any]] = []
        for dataset_id, dataset_name in enumerate(self.dataset_names):
            dataset_dir = self.root_dir / dataset_name
            depth_files = self._modality_files(dataset_dir, "depth_vel")
            if depth_files is None:
                raise FileNotFoundError(f"OpenFWI dataset {dataset_name!r} is missing required depth_vel files.")

            modality_files: dict[str, list[Path] | None] = {"depth_vel": depth_files}
            for modality in STANDARD_MODALITIES:
                if modality == "well_log":
                    continue
                files = self._modality_files(dataset_dir, modality)
                if files is not None and len(files) != len(depth_files):
                    raise ValueError(
                        f"OpenFWI {dataset_name}/{modality} has {len(files)} files, "
                        f"but depth_vel has {len(depth_files)} files."
                    )
                modality_files[modality] = files

            for sample_index, depth_path in enumerate(depth_files):
                record: dict[str, Any] = {
                    "dataset_name": dataset_name,
                    "dataset_id": dataset_id,
                    "sample_index": sample_index,
                    "paths": {"depth_vel": depth_path},
                }
                for modality in STANDARD_MODALITIES:
                    if modality == "well_log":
                        continue
                    files = modality_files[modality]
                    record["paths"][modality] = None if files is None else files[sample_index]
                records.append(record)
        return records

    def _build_lmdb_records(self) -> list[dict[str, Any]]:
        """Build an index over per-subset LMDB stores."""
        records: list[dict[str, Any]] = []
        required = set(self.required_modalities)
        for dataset_id, dataset_name in enumerate(self.dataset_names):
            reader = OpenFWILMDBReader(openfwi_lmdb_path(self.lmdb_root, dataset_name))
            missing = sorted(required.difference(reader.manifest["modalities"]))
            if missing:
                raise KeyError(f"OpenFWI LMDB {dataset_name} is missing required modalities: {missing}")
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

    @staticmethod
    def _legacy_val_fraction_to_split_fractions(val_fraction: float) -> tuple[float, float, float]:
        """Translate the older `val_fraction` option into train/val/test fractions.

        This preserves backward compatibility for configs that still pass
        `val_fraction`, but new configs should use `split_fractions` directly.
        """
        held_out = max(float(val_fraction), 0.0)
        train_fraction = max(1.0 - 2.0 * held_out, 0.0)
        return _validate_split_fractions((train_fraction, held_out, held_out))

    @staticmethod
    def _select_split_then_filter(
        records: list[dict[str, Any]],
        split: str,
        split_fractions: tuple[float, float, float],
        split_seed: int,
        post_split_datasets: list[str] | tuple[str, ...] | None,
    ) -> list[dict[str, Any]]:
        selected = OpenFWIBGDataset._select_split(records, split, split_fractions, split_seed)
        if post_split_datasets is None:
            return selected
        allowed = set(post_split_datasets)
        return [record for record in selected if record["dataset_name"] in allowed]

    @staticmethod
    def _select_split(
        records: list[dict[str, Any]],
        split: str,
        split_fractions: tuple[float, float, float],
        split_seed: int,
    ) -> list[dict[str, Any]]:
        """Create a random but reproducible train/val/test split view.

        For the default 7:2:1 split, samples are assigned by stable index order:
            train -> 70% of a fixed random permutation
            val   -> 20% of a fixed random permutation
            test  -> 10% of a fixed random permutation

        DataLoader shuffling is still controlled separately. The train loader
        uses `shuffle=True`, so training order is randomized after this split.
        """
        shuffled_records = [records[i] for i in _shuffled_indices(len(records), split_seed)]
        if split == "all":
            return shuffled_records
        total = len(shuffled_records)
        if total <= 1:
            return shuffled_records
        train_fraction, val_fraction, _ = split_fractions
        train_end = int(round(total * train_fraction))
        val_end = train_end + int(round(total * val_fraction))
        train_end = max(1, min(train_end, total))
        val_end = max(train_end, min(val_end, total))
        if val_end >= total and total >= 3:
            val_end = total - 1
        if train_end >= val_end and total >= 3:
            train_end = val_end - 1
        if split == "train":
            return shuffled_records[:train_end]
        if split in {"val", "valid"}:
            return shuffled_records[train_end:val_end]
        if split == "test":
            return shuffled_records[val_end:]
        raise ValueError("OpenFWIBGDataset split must be one of 'train', 'val', 'valid', 'test', or 'all'.")

    def __len__(self) -> int:
        return len(self.records)

    def close(self) -> None:
        """Release LMDB file handles held by this dataset."""
        for reader in self._lmdb_readers.values():
            reader.close()
        self._lmdb_readers.clear()

    def _normalize(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        return _profile_normalize_tensor(
            tensor,
            key,
            profile=self.normalization_profile,
            mode=self.use_normalize,
            clamp=self.normalize_clamp,
        )

    def _load_raw_tensor(self, key: str, path: Path) -> torch.Tensor:
        """Load a raw tensor from `.npy` and standardize its shape."""
        tensor = torch.from_numpy(np.asarray(np.load(path, allow_pickle=False))).to(torch.float32)
        return _standardize_tensor(key, tensor, self.target_shape, context="OpenFWIBGDataset")

    def _load_lmdb_raw_tensor(self, key: str, record: dict[str, Any]) -> torch.Tensor:
        """Load a raw tensor from LMDB and standardize its shape."""
        reader = self._lmdb_readers[record["dataset_name"]]
        tensor = torch.from_numpy(np.asarray(reader.get(record["sample_index"], key))).to(torch.float32)
        return _standardize_tensor(key, tensor, self.target_shape, context="OpenFWIBGDataset")

    def _load_tensor(self, key: str, path: Path) -> torch.Tensor:
        tensor = self._load_raw_tensor(key, path)
        return self._normalize(key, tensor)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        paths = record["paths"]
        # `depth_vel` is the source of truth for both the model input and the
        # dynamic well synthesis path below.
        if self.storage_backend == "lmdb":
            raw_depth_vel = self._load_lmdb_raw_tensor("depth_vel", record)
        else:
            raw_depth_vel = self._load_raw_tensor("depth_vel", paths["depth_vel"])
        depth_vel = self._normalize("depth_vel", raw_depth_vel)
        raw_well_log, well_mask = _build_random_well(
            raw_depth_vel,
            idx=int(record["dataset_id"]) * 1_000_000 + int(record["sample_index"]),
            well_count_range=self.well_count_range,
            well_seed=self.well_seed,
            well_random=self.well_random,
        )
        well_log = _normalize_sparse_well_log(well_log=raw_well_log, well_mask=well_mask,
                                              use_normalize=self.use_normalize,
                                              normalize_max_min=OPENFWI_NORMALIZE_MAX_MIN,
                                              normalization_profile=self.normalization_profile,
                                              normalize_clamp=self.normalize_clamp)
        item: dict[str, torch.Tensor] = {"depth_vel": depth_vel}
        mask_values: list[float] = []
        quality_values: list[float] = []

        for modality in STANDARD_MODALITIES:
            if modality == "well_log":
                item["well_log"] = well_log
                item["well_mask"] = well_mask
                mask_values.append(_has_well_observation(well_mask))
                quality_values.append(mask_values[-1])
                continue

            path = paths.get(modality)
            if self.storage_backend == "lmdb":
                item[modality] = self._normalize(modality, self._load_lmdb_raw_tensor(modality, record))
                mask_values.append(1.0)
                quality_values.append(1.0)
            elif path is None:
                item[modality] = torch.zeros_like(depth_vel)
                mask_values.append(0.0)
                quality_values.append(0.0)
            else:
                item[modality] = self._load_tensor(modality, path)
                mask_values.append(1.0)
                quality_values.append(1.0)

        # `modality_mask` says whether a modality exists at all.
        # `modality_quality` is kept separate for future quality-aware weighting.
        item["modality_mask"] = torch.tensor(mask_values, dtype=torch.float32)
        item["modality_quality"] = torch.tensor(quality_values, dtype=torch.float32)
        item["dataset_id"] = torch.tensor(float(record["dataset_id"]), dtype=torch.float32)
        item["sample_index"] = torch.tensor(float(record["sample_index"]), dtype=torch.float32)
        return item


class SyntheticBGDataset(Dataset):
    """Tiny deterministic dataset for smoke tests and wiring checks.

    This is not a real training dataset. It only exists to verify that the
    model, collate function, and dataloading path agree on the sample schema.
    """

    def __init__(self, length: int = 8, image_shape: tuple[int, int, int] = (1, 70, 70), seed: int = 1234) -> None:
        self.length = int(length)
        self.image_shape = image_shape
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(self.seed + int(idx))
        depth_vel = torch.randn(*self.image_shape, generator=gen).clamp(-2.0, 2.0) / 2.0
        smooth = torch.nn.functional.avg_pool2d(depth_vel.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)
        migrated = depth_vel - smooth + 0.05 * torch.randn(*self.image_shape, generator=gen)
        horizon = (torch.abs(migrated) > migrated.abs().mean()).to(torch.float32)
        rms = smooth
        well = torch.zeros_like(depth_vel)
        well_mask = torch.zeros_like(depth_vel)
        well[..., self.image_shape[-1] // 2] = depth_vel[..., self.image_shape[-1] // 2]
        well_mask[..., self.image_shape[-1] // 2] = 1.0
        modality_mask = torch.ones(len(STANDARD_MODALITIES), dtype=torch.float32)
        modality_quality = torch.ones(len(STANDARD_MODALITIES), dtype=torch.float32)
        well_index = STANDARD_MODALITIES.index("well_log")
        modality_mask[well_index] = _has_well_observation(well_mask)
        modality_quality[well_index] = modality_mask[well_index]
        return {
            "depth_vel": depth_vel,
            "migrated_image": migrated,
            "horizon": horizon,
            "rms_vel": rms,
            "well_log": well,
            "well_mask": well_mask,
            "modality_mask": modality_mask,
            "modality_quality": modality_quality,
        }

