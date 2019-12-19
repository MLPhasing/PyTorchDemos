## Subset Data

Sampling a subset of data with target remapping.
If you only need to sample a subset of data without target transform, consider using [`SubsetRandomSampler`](https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#SubsetRandomSampler)

An example on how to use the code:

```python
train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform)
target_mapping = {0:0, 1:0, 2:0, 3:0, 4:0, 5:1, 6:2, 7:3, 8:4, 9:0}
data_size = 100
subdata = SubsetData(train_set, data_size, target_mapping = target_mapping, random_seed = 7 )
print(len(subdata))
sub_loader = torch.utils.data.DataLoader(subdata, batch_size = 10, shuffle=True)
for x,y in sub_loader:
    print(len(x), y)
```
