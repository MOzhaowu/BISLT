"""Validation-buffer utilities for uncertainty-aware model updates."""

import math

import numpy as np


def pose_signature(pose, decimals=5):
    return tuple(np.asarray(pose, dtype=np.float64).round(decimals).ravel())


def rotation_distance_deg(left, right):
    relative = np.asarray(left)[:3, :3] @ np.asarray(right)[:3, :3].T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def translation_distance(left, right):
    return float(np.linalg.norm(np.asarray(left)[:3, 3] - np.asarray(right)[:3, 3]))


def mask_quality(mask):
    array = np.asarray(mask)
    foreground = array[3] if array.ndim == 3 and array.shape[0] == 4 else array
    ratio = float(np.count_nonzero(foreground > 0.5)) / foreground.size
    return max(0.0, 1.0 - abs(ratio - 0.35) / 0.35)


class ValidationBuffer:
    def __init__(self, capacity=32):
        self.capacity = int(capacity)
        self._records = []
        self._signatures = set()

    def add(self, masks, poses, intrinsics):
        for mask, pose, intrinsic in zip(masks, poses, intrinsics):
            pose = np.asarray(pose, dtype=np.float32)
            signature = pose_signature(pose)
            if signature in self._signatures:
                continue
            self._records.append({
                'mask': np.asarray(mask, dtype=np.float32).copy(),
                'pose': pose.copy(),
                'intrinsic': np.asarray(intrinsic, dtype=np.float32).copy(),
                'signature': signature,
            })
            self._signatures.add(signature)
        while len(self._records) > self.capacity:
            removed = self._records.pop(0)
            self._signatures.remove(removed['signature'])

    def select(self, count, excluded_signatures=(), rotation_weight=1.0,
               translation_weight=100.0, quality_weight=0.25):
        excluded = set(excluded_signatures)
        candidates = [r for r in self._records if r['signature'] not in excluded]
        selected = []
        while candidates and len(selected) < int(count):
            def score(record):
                quality = quality_weight * mask_quality(record['mask'])
                references = selected
                if not references:
                    return quality
                diversity = min(
                    rotation_weight * rotation_distance_deg(record['pose'], ref['pose']) / 180.0
                    + translation_weight * translation_distance(record['pose'], ref['pose'])
                    for ref in references
                )
                return diversity + quality

            chosen_index = max(
                range(len(candidates)), key=lambda index: score(candidates[index])
            )
            chosen = candidates.pop(chosen_index)
            selected.append(chosen)
        return selected

    def __len__(self):
        return len(self._records)
