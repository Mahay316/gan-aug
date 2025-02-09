import json
import random
import os.path

import torch
import numpy as np
import pickle

from torch.utils.data import Dataset, DataLoader


class TrafficDataset(Dataset):
    def __init__(self, data_dir, index_file='dataset.idx', transform=None):
        self.data_dir = data_dir
        self.transform = transform

        with open(os.path.join(data_dir, index_file), 'rb') as f:
            self.dataset_index = pickle.load(f)
            self.index_to_filename = pickle.load(f)
            self.index_to_label = pickle.load(f)

        self.file_pool = []
        for filename in self.index_to_filename:
            self.file_pool.append(
                open(os.path.join(data_dir, filename), 'r')
            )

    def calculate_file_and_offset(self, idx):
        file_idx = 0

        for file in self.dataset_index:
            if idx < len(file):
                break  # idx falls into this file
            else:
                idx -= len(file)
                file_idx += 1

        return file_idx, self.dataset_index[file_idx][idx]

    def get_sample_count(self):
        sample_counts = [len(x) for x in self.dataset_index]
        res_sorted = [x for _, x in sorted(zip(self.index_to_label, sample_counts))]
        return res_sorted

    def get_labels(self):
        ret = []
        for label, dataset in zip(self.index_to_label, self.dataset_index):
            ret.extend([label] * len(dataset))
        return ret

    def get_subset_labels(self, subset: torch.utils.data.Subset):
        ret = np.zeros(len(subset.indices), dtype=np.int32)
        for i, idx in enumerate(subset.indices):
            file_idx, _ = self.calculate_file_and_offset(idx)
            ret[i] = self.index_to_label[file_idx]

        return ret

    def __len__(self):
        return sum([len(x) for x in self.dataset_index])

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        file_idx, offset = self.calculate_file_and_offset(idx)
        target_file = self.file_pool[file_idx]
        target_file.seek(offset)

        trace = json.loads(target_file.readline())
        sample = (trace[0], trace[1])

        return sample


def should_drop(prob) -> bool:
    granularity = 1000
    return random.randint(0, granularity - 1) < prob * granularity


def get_my_collate(drop_prob=0, drop_class=None):
    def my_collate_fn(data):
        traces = []
        labels = []

        for sample in data:
            if sample[1] != drop_class or not should_drop(drop_prob):
                traces.append(torch.tensor(sample[0]))
                labels.append(sample[1])

        return traces, labels

    return my_collate_fn


if __name__ == '__main__':
    root_dir = 'D:/traffic_dataset/interval/normalized'
    cd = TrafficDataset(root_dir)

    print(cd.get_sample_count())

    # train, val, test = torch.utils.data.random_split(cd, [0.8, 0.1, 0.1])
    # dl = DataLoader(test, batch_size=32, shuffle=True, collate_fn=get_my_collate(1, 1))
    #
    # print(len(dl))
    #
    # for batch_idx, samples in enumerate(dl):
    #     print(samples[1])
    # traces, labels = samples
    # for trace in traces:
    #     print(trace)
    # break
