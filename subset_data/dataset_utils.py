import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from typing import Dict
from collections import defaultdict
import itertools as it
import warnings

def split_dataset(dataset: Dataset, train_frac: float = 0.9, random_seed: int = None):
    random_seed = random_seed if random_seed else np.random.randint(1<<10)
    np.random.seed(random_seed)
    indicies = np.arange(len(dataset))
    np.random.shuffle(indicies)
    split = int(len(dataset)*train_frac)
    return Subset(dataset, indicies[:split]), Subset(dataset, indicies[split:])


class TargetTransformDataset(Dataset):
    """
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        target_mapping = {0:0, 1:0, 2:0, 3:0, 4:0, 5:1, 6:2, 7:3, 8:4, 9:0}
        data = TargetTransformDataset(train_set, target_mapping = target_mapping)

        target_mapping is a dictionary to map an old index to new index
    """

    def __init__(self, dataset: Dataset, *_, target_mapping: Dict):
        self.dataset = dataset
        self.target_mapping = target_mapping

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        y = self.target_mapping[y]
        return x, y 
    
    def __len__(self):
        return len(self.dataset)
        

class StatefulDataSampler():
    """
    Example,
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        target_mapping = {0:0, 1:0, 2:0, 3:0, 4:0, 5:1, 6:2, 7:3, 8:4, 9:0}
        data_size = 100
        subdata = StatefulSubsetDataset(train_set, data_size, random_seed = 7 )
        print(len(subdata))
        sub_loader = torch.utils.data.DataLoader(subdata, batch_size = 10, shuffle=True)
        for x,y in sub_loader:
            print(len(x), y)

    """

    def __init__(self, dataset: Dataset, random_seed: int = None):
        """
            dataset : dataset[i] returns a tuple of (x[i], y[i])
            random_seed : int
        """
        self.rndseed = random_seed if random_seed else np.random.randint(1<<20)
        np.random.seed(self.rndseed)
        self.state = defaultdict(dict)
        self.total_sz, self.num_cls = 0, 0

        cls2idx = defaultdict(list)
        for i, (_, y) in enumerate(dataset):
            cls2idx[y].append(i)

        for k, v in cls2idx.items():
            tmp = np.asarray(v)
            np.random.shuffle(tmp)
            self.state[k] = {'idx': 0, 'data': np.copy(tmp)}
            self.total_sz += len(tmp)
            self.num_cls += 1

    def __len__(self):
        return sum(v['idx'] for v in self.state.values())

    def __validate_budget(self, data_size: int):
        if data_size > self.total_sz - len(self):
            raise RuntimeError(f"insufficient data, trying to get {data_size}\
                               but with {self.total_sz-len(self)} remains")

    def __distribute_budget(self, data_size):
        per_cls_sz = data_size//self.num_cls
        remain_sz = data_size - per_cls_sz * self.num_cls
        res = dict(zip(self.state.keys(), it.repeat(per_cls_sz)))
        if remain_sz > 0:
            remains = np.random.choice(self.num_cls, size=remain_sz,
                                       replace=False)
            remains = set(remains)
            for i, (cls_id, _) in enumerate(self.state.items()):
                if i in remains:
                    res[cls_id] += 1

        return res

    def get_samples(self):
        return np.concatenate(list(v['data'][:v['idx']] \
                               for v in self.state.values()))

    def add_samples(self, data_size: int or dict):
        """
            if data_size is int, 
                * distribute the budget to each classes evenly at random
                * unless some classes can not fulfill the random requirement 
            if data_size is a dictionary,
                * distribute the budget accordingly
        """

        if isinstance(data_size, int):
            self.__validate_budget(data_size)
            data_size = self.__distribute_budget(data_size)
        elif isinstance(data_size, dict):
            assert set(self.state.keys()) == set(data_size.keys())
            self.__validate_budget(sum(data_size.values()))
        else:
            raise RuntimeError(f"data_size type {type(data_size)}\
                               is not supported")

        unfulfilled_sz = 0 

        for k, v in data_size.items():
            if self.state[k]['idx'] + v < len(self.state[k]['data']):
                self.state[k]['idx'] += v
            else:
                tmp = len(self.state[k]['data']) - self.state[k]['idx']
                unfulfilled_sz += v-tmp
                self.state[k]['idx'] = len(self.state[k]['data'])

        if unfulfilled_sz > 0:
            warnings.warn(f"budget cannot be fullfilled with remaining\
                           samples: {unfulfilled_sz}", RuntimeWarning)
            while unfulfilled_sz > 0:
                for v in self.state.values():
                    if v['idx'] < len(v['data']):
                        v['idx'] += 1
                        unfulfilled_sz -= 1
                        if unfulfilled_sz == 0: break

        return True
