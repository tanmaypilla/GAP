import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import sys

sys.path.extend(['../'])

class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, window_size=64, normalization=False, debug=False, use_mmap=False, bone=False, vel=False):
        """
        Feeder for Hockey Skating Actions Dataset
        """
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.bone = bone
        self.vel = vel
        
        self.label = [] # Will be populated in load_data
        self.load_data()
        
        # --- FIX 1: Removed self.data.shape crash ---
        print(f"[DEBUG] Loading data from: {self.data_path}")
        print(f"[DEBUG] Total Valid Samples: {len(self.valid_indices)}")
        
        # Safe coordinate check (load first sample to peek)
        if len(self.valid_indices) > 0:
            first_data, _, _ = self.__getitem__(0)
            print(f"[DEBUG] Sample Coordinate (First frame, First joint): {first_data[:, 0, 0, 0]}")

    def load_data(self):
        # Load the pickle file
        try:
            with open(self.data_path, 'rb') as f:
                self.file_content = pickle.load(f)
        except Exception as e:
            print(f"Error loading data from {self.data_path}: {e}")
            sys.exit(1)

        if isinstance(self.file_content, dict) and 'annotations' in self.file_content:
            self.annotations = self.file_content['annotations']
        else:
            self.annotations = self.file_content

        self.valid_indices = []
        self.label = [] 
        
        print(f"[INFO] Filtering data (Total raw samples: {len(self.annotations)})...")
        
        for i, sample in enumerate(self.annotations):
            if 'keypoint' in sample and sample['keypoint'] is not None:
                self.valid_indices.append(i)
                self.label.append(sample.get('label', -1))

        if self.debug:
            self.valid_indices = self.valid_indices[:100]
            self.label = self.label[:100]

    def __len__(self):
        return len(self.valid_indices)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        actual_index = self.valid_indices[index]
        sample = self.annotations[actual_index]
        
        # Raw Data: (1, T, 20, 2)
        data_numpy = np.array(sample['keypoint']) 
        
        # Transpose to Model Format: (C, T, V, M)
        data_numpy = np.transpose(data_numpy, (3, 1, 2, 0))

        label = sample.get('label', -1)

        # --- NORMALIZATION (Mid-Hip Center + SCALING) ---
        # Applied before padding so zero-padded frames remain true zeros
        if self.normalization:
            # 1. Center the skeleton
            right_hip = data_numpy[:, 0, 5, 0]
            left_hip  = data_numpy[:, 0, 6, 0]
            origin = (right_hip + left_hip) / 2.0
            origin = origin[:, np.newaxis, np.newaxis, np.newaxis]
            data_numpy = data_numpy - origin

            # 2. Scale to approx [-1, 1] range
            data_numpy = data_numpy / 1000.0

        # --- BONE MODALITY ---
        # Applied before padding so padded zeros don't produce spurious bone vectors
        if self.bone:
            hockey_pairs = (
                (2, 1), (2, 0), (0, 1), (3, 4), (3, 5), (4, 6), (5, 6),
                (3, 7), (7, 10), (4, 9), (9, 8), (5, 11), (11, 14),
                (14, 15), (6, 12), (12, 13), (13, 16), (17, 18), (18, 19)
            )
            bone_data = np.zeros_like(data_numpy)
            for v1, v2 in hockey_pairs:
                bone_data[:, :, v1, :] = data_numpy[:, :, v1, :] - data_numpy[:, :, v2, :]
            data_numpy = bone_data

        # --- VELOCITY MODALITY ---
        # Applied before padding so padded zeros don't produce spurious velocities
        if self.vel:
            vel_data = np.zeros_like(data_numpy)
            vel_data[:, :-1, :, :] = data_numpy[:, 1:, :, :] - data_numpy[:, :-1, :, :]
            vel_data[:, -1, :, :] = vel_data[:, -2, :, :]
            data_numpy = vel_data

        # Resize temporal length to window_size (64) via zero-padding
        data_numpy = self._resize_temporal(data_numpy, self.window_size)

        return data_numpy, label, actual_index

    def top_k(self, score, top_k):
        if len(self.label) == 0:
            return 0.0
        min_len = min(len(self.label), len(score))
        rank = score.argsort()
        hit_top_k = [self.label[i] in rank[i, -top_k:] for i in range(min_len)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
    
    def _resize_temporal(self, data, target_length):
        C, T, V, M = data.shape
        if T == target_length:
            return data
        if T > target_length:
            return data[:, :target_length, :, :]
        new_data = np.zeros((C, target_length, V, M), dtype=data.dtype)
        new_data[:, :T, :, :] = data
        return new_data
