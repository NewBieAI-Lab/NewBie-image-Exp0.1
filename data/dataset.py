from abc import ABC, abstractmethod
import copy
import json
import logging
import os
from pathlib import Path
import random
from time import sleep
import traceback
import warnings

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import torch.distributed as dist
from torch.utils.data import Dataset
import yaml
from PIL import Image

from .data_reader import read_general

logger = logging.getLogger(__name__)


class DataBriefReportException(Exception):
    def __init__(self, message=None):
        self.message = message

    def __str__(self):
        return f"{self.__class__}: {self.message}"


class DataNoReportException(Exception):
    def __init__(self, message=None):
        self.message = message

    def __str__(self):
        return f"{self.__class__}: {self.message}"


class ItemProcessor(ABC):
    @abstractmethod
    def process_item(self, data_item, training_mode=False):
        raise NotImplementedError


class MyDataset(Dataset):
    def __init__(self, config_path, item_processor: ItemProcessor, cache_on_disk=False):
        logger.info(f"read dataset config from {config_path}")
        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        logger.info("DATASET CONFIG:")
        logger.info(self.config)

        self.config_path = config_path

        self.cache_on_disk = cache_on_disk
        if self.cache_on_disk:
            cache_dir = self._get_cache_dir(config_path)
            if dist.get_rank() == 0:
                self._collect_annotations_and_save_to_cache(cache_dir)
            dist.barrier()
            ann, group_indice_range = self._load_annotations_from_cache(cache_dir)
        else:
            cache_dir = None
            ann, group_indice_range = self._collect_annotations()

        self.ann = ann
        self.group_indices = {key: list(range(val[0], val[1])) for key, val in group_indice_range.items()}

        logger.info(f"total length: {len(self)}")

        self.item_processor = item_processor

        self.sample_bucket = None
        self.bucket_indices = None

    def __len__(self):
        return len(self.ann)

    def _collect_annotations(self):
        group_ann = {}
        for meta in self.config["META"]:
            meta_path, meta_type = meta["path"], meta.get("type", "default")
            meta_ext = os.path.splitext(meta_path)[-1]
            if meta_ext == ".json":
                with open(meta_path, "r") as json_file:
                    f = json_file.read()
                    meta_l = json.loads(f)
            elif meta_ext == ".jsonl":
                meta_l = []
                with open(meta_path) as f:
                    for i, line in enumerate(f):
                        try:
                            meta_l.append(json.loads(line))
                        except json.decoder.JSONDecodeError as e:
                            logger.error(f"Error decoding the following jsonl line ({i}):\n{line.rstrip()}")
                            raise e
            else:
                raise NotImplementedError(
                    f'Unknown meta file extension: "{meta_ext}". '
                    f"Currently, .json, .jsonl are supported. "
                    "If you are using a supported format, please set the file extension so that the proper parsing "
                    "routine can be called."
                )
            logger.info(f"{meta_path}, type{meta_type}: len {len(meta_l)}")
            if "ratio" in meta:
                random.seed(0)
                meta_l = random.sample(meta_l, int(len(meta_l) * meta["ratio"]))
                logger.info(f"sample (ratio = {meta['ratio']}) {len(meta_l)} items")
            if "root" in meta:
                for item in meta_l:
                    for path_key in ["path", "image_url", "image", "image_path"]:
                        if path_key in item:
                            item[path_key] = os.path.join(meta["root"], item[path_key])
            if meta_type not in group_ann:
                group_ann[meta_type] = []
            group_ann[meta_type] += meta_l

        ann = sum(list(group_ann.values()), start=[])

        group_indice_range = {}
        start_pos = 0
        for meta_type, meta_l in group_ann.items():
            group_indice_range[meta_type] = [start_pos, start_pos + len(meta_l)]
            start_pos = start_pos + len(meta_l)

        return ann, group_indice_range

    def _collect_annotations_and_save_to_cache(self, cache_dir):
        if (Path(cache_dir) / "data.h5").exists() and (Path(cache_dir) / "ready").exists():
            warnings.warn(
                f"Use existing h5 data cache: {Path(cache_dir)}\n"
                f"Note: if the actual data defined by the data config has changed since your last run, "
                f"please delete the cache manually and re-run this experiment, or the data actually used "
                f"will not be updated"
            )
            return

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        ann, group_indice_range = self._collect_annotations()

        serialized_ann = [json.dumps(_) for _ in ann]
        logger.info(f"start to build data cache to: {Path(cache_dir)}")
        with h5py.File(Path(cache_dir) / "data.h5", "w") as file:
            dt = h5py.vlen_dtype(str)
            h5_ann = file.create_dataset("ann", (len(serialized_ann),), dtype=dt)
            h5_ann[:] = serialized_ann
            file.create_dataset("group_indice_range", data=json.dumps(group_indice_range))
        with open(Path(cache_dir) / "ready", "w") as f:
            f.write("ready")
        logger.info("data cache built")

    @staticmethod
    def _get_cache_dir(config_path):
        config_identifier = config_path
        disallowed_chars = ["/", "\\", ".", "?", "!"]
        for _ in disallowed_chars:
            config_identifier = config_identifier.replace(_, "-")
        cache_dir = f"./accessory_data_cache/{config_identifier}"
        return cache_dir

    @staticmethod
    def _load_annotations_from_cache(cache_dir):
        while not (Path(cache_dir) / "ready").exists():
            assert dist.get_rank() != 0
            sleep(1)
        cache_file = h5py.File(Path(cache_dir) / "data.h5", "r")
        annotations = cache_file["ann"]
        group_indice_range = json.loads(cache_file["group_indice_range"].asstr()[()])
        return annotations, group_indice_range

    def get_item_func(self, index):
        data_item = self.ann[index]
        if self.cache_on_disk:
            data_item = json.loads(data_item)
        else:
            data_item = copy.deepcopy(data_item)
        return self.item_processor.process_item(data_item, training_mode=True)

    def __getitem__(self, index):
        try:
            return self.get_item_func(index)
        except Exception as e:
            if isinstance(e, DataNoReportException):
                pass
            elif isinstance(e, DataBriefReportException):
                logger.info(e)
            else:
                logger.info(
                    f"Item {index} errored, annotation:\n"
                    f"{self.ann[index]}\n"
                    f"Error:\n"
                    f"{traceback.format_exc()}"
                )
            for group_name, indices_this_group in self.group_indices.items():
                if indices_this_group[0] <= index <= indices_this_group[-1]:
                    if index == indices_this_group[0]:
                        new_index = indices_this_group[-1]
                    else:
                        new_index = index - 1
                    return self[new_index]
            raise RuntimeError

    def assign_buckets(self, crop_size_list, random_top_k: int = 1):
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        cache_dir = self._get_cache_dir(self.config_path)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        size_cache_path = Path(cache_dir) / "sizes.h5"
        ready_path = Path(cache_dir) / "sizes.ready"

        num_samples = len(self.ann)

        valid_cache = False
        if size_cache_path.exists() and ready_path.exists():
            try:
                with h5py.File(size_cache_path, "r") as f:
                    sizes = f["sizes"]
                    if sizes.shape[0] == num_samples and sizes.shape[1] == 2:
                        valid_cache = True
            except Exception:
                valid_cache = False

        if not valid_cache:
            chunk = (num_samples + world_size - 1) // world_size
            start = rank * chunk
            end = min(num_samples, (rank + 1) * chunk)
            local_n = max(0, end - start)
            local_path = Path(cache_dir) / f"sizes_rank{rank}.npy"

            if local_n > 0 and not local_path.exists():
                logger.info(f"assign_buckets rank{rank}: scanning [{start}, {end}) / {num_samples}")
                arr = np.zeros((local_n, 2), dtype="int32")
                for loc, idx in enumerate(range(start, end)):
                    if self.cache_on_disk:
                        try:
                            item = json.loads(self.ann[idx])
                        except Exception:
                            item = None
                    else:
                        item = self.ann[idx]

                    w = h = 0
                    if isinstance(item, dict):
                        img_path = None
                        for key in ["path", "image_path", "image", "image_url"]:
                            if key in item:
                                img_path = item[key]
                                break
                        if img_path is not None:
                            try:
                                with Image.open(read_general(img_path)) as im:
                                    w, h = im.size
                            except Exception:
                                w = h = 0

                    arr[loc, 0] = int(w)
                    arr[loc, 1] = int(h)

                    if (loc + 1) % 100000 == 0:
                        logger.info(f"assign_buckets rank{rank}: scanned {loc + 1} / {local_n} images")
                np.save(local_path, arr)

            if world_size > 1 and dist.is_initialized():
                dist.barrier()

            if rank == 0:
                logger.info(f"assign_buckets: merging size chunks into {size_cache_path}")
                full = np.zeros((num_samples, 2), dtype="int32")
                chunk = (num_samples + world_size - 1) // world_size
                for r in range(world_size):
                    r_start = r * chunk
                    r_end = min(num_samples, (r + 1) * chunk)
                    r_n = max(0, r_end - r_start)
                    if r_n <= 0:
                        continue
                    part_path = Path(cache_dir) / f"sizes_rank{r}.npy"
                    part = np.load(part_path)
                    full[r_start:r_end] = part[:r_n]
                with h5py.File(size_cache_path, "w") as f:
                    f.create_dataset("sizes", data=full, dtype="int32")
                with open(ready_path, "w") as f:
                    f.write("ready")
                logger.info("assign_buckets: size cache build finished")

            if world_size > 1 and dist.is_initialized():
                dist.barrier()

        while not ready_path.exists():
            sleep(5)

        self.sample_bucket = [-1] * num_samples
        self.bucket_indices = {}

        with h5py.File(size_cache_path, "r") as f:
            sizes = f["sizes"]
            crop_arr = np.array(crop_size_list, dtype=np.float32)  # (B, 2)
            cw = crop_arr[:, 0]
            ch = crop_arr[:, 1]
            top_k = max(1, random_top_k)
            chunk_size = 200000

            for start in range(0, num_samples, chunk_size):
                end = min(num_samples, start + chunk_size)
                wh = sizes[start:end].astype(np.float32)
                w = wh[:, 0]
                h = wh[:, 1]
                valid_mask = (w > 0) & (h > 0)
                if not np.any(valid_mask):
                    continue

                idx_valid = np.nonzero(valid_mask)[0]
                wv = w[idx_valid][:, None]
                hv = h[idx_valid][:, None]

                rw = cw[None, :] / wv
                rh = ch[None, :] / hv
                min_ratio = np.minimum(rw, rh)
                max_ratio = np.maximum(rw, rh)
                score = np.where(max_ratio > 0, min_ratio / max_ratio, 0.0)

                if top_k == 1:
                    best = np.argmax(score, axis=1)
                else:
                    idx_topk = np.argpartition(score, -top_k, axis=1)[:, -top_k:]
                    rand_idx = np.random.randint(0, idx_topk.shape[1], size=idx_topk.shape[0])
                    best = idx_topk[np.arange(idx_topk.shape[0]), rand_idx]

                global_idx = start + idx_valid
                for gi, b in zip(global_idx, best):
                    b_int = int(b)
                    self.sample_bucket[gi] = b_int
                    if b_int not in self.bucket_indices:
                        self.bucket_indices[b_int] = []
                    self.bucket_indices[b_int].append(int(gi))

        assigned = sum(1 for b in self.sample_bucket if b != -1)
        logger.info(
            f"assign_buckets: assigned {assigned} / {num_samples} samples into "
            f"{len(self.bucket_indices) if self.bucket_indices is not None else 0} buckets."
        )

    def groups(self):
        return list(self.group_indices.values())


