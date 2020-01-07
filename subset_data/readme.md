# PyTorch Data Utilities

## Stateful Data Sampler
An example on how to use the code:

```python
import itertools as its
import torchvision
from torch.utils.data import Subset
from dataset_utils import StatefulDataSampler

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform)
train_sampler = StatefulDataSampler(tt_train_set, random_seed=7)
train_sampler.add_samples( 100 ) # add 100 samples
train_data_set = Subset(train_set, train_sampler.get_samples())
## do some training on train_data_set

## add 20 samples per class on top of previous sampled data
train_sampler.add_samples(dict(zip(range(10), its.repeat(20)))) 

train_data_set = Subset(train_set, train_sampler.get_samples())
## do some more training
```

If you only need to sample a subset of data once, consider using PyTorch
[`SubsetRandomSampler`](https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#SubsetRandomSampler)

## Target Transform 

```
import torchvision
from dataset_utils import TargetTransformDataset

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
target_mapping = {0:0, 1:0, 2:0, 3:0, 4:0, 5:1, 6:2, 7:3, 8:4, 9:0}
data_set = TargetTransformDataset(train_set, target_mapping = target_mapping)
##then, loader = DataLoader(data_set, ...), and so on
```
