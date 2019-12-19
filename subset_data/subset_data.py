import numpy as np
from torch.utils.data import Dataset
class SubsetData(Dataset):
    """
    Example,
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
        target_mapping = {0:0, 1:0, 2:0, 3:0, 4:0, 5:1, 6:2, 7:3, 8:4, 9:0}
        data_size = 100
        subdata = SubsetData(train_set, data_size, target_mapping = target_mapping, random_seed = 7 )
        print(len(subdata))
        sub_loader = torch.utils.data.DataLoader(subdata, batch_size = 10, shuffle=True)
        for x,y in sub_loader:
            print(len(x), y)
    
    """
    def __init__(self, dataset : torch.utils.data.Dataset, data_size : int, target_mapping=None, random_seed=None):
        """
        
            dataset : dataset[i] returns a tuple of (x[i], y[i]) such that y[i] \in target_mapping.keys()
            data_size : randomly sampled subset of data with data_size
            target_mapping : map original class to newly defined class. 
        """
        if data_size <= 0 or data_size > len(dataset):
            raise ValueError("invalid data_size", data_size)
            
        self.random_seed, self.target_mapping = random_seed, target_mapping
        if random_seed:
            tmp_state = np.random.get_state()
            np.random.seed(random_seed)
        self.rnd_state = np.random.get_state()
        
        self.data_ptr = dataset
        
        cls2idx = {}
        for i, (x,y) in enumerate(dataset):
            if target_mapping:
                y = target_mapping[y]
            if y not in cls2idx:
                cls2idx[y] = []
            cls2idx[y].append(i)
        
        per_cls_sz = [data_size//len(cls2idx)]*len(cls2idx)
        if data_size > sum(per_cls_sz):
            extra = data_size - sum(per_cls_sz)
            idx_ = np.asarray(list(range(len(cls2idx))))
            np.random.shuffle(idx_)
            for i in range(extra):
                per_cls_sz[idx_[i]] += 1
        
        self.selected_indices = []
        for cls_, sz_ in zip(cls2idx.keys(), per_cls_sz):
            v_ = np.asarray(cls2idx[cls_])
            np.random.shuffle(v_)
            self.selected_indices.extend(v_[:sz_])
        
        # recover rnd state 
        if random_seed: np.random.set_state(tmp_state)
        
    def __getitem__(self, idx):
        x, y = self.data_ptr[self.selected_indices[idx]]
        if self.target_mapping: y = self.target_mapping[y]
        return x,y 
    
    def __len__(self):
        return len(self.selected_indices)
