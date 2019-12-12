import argparse
import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.device = torch.cuda.current_device()
        self.head = nn.Linear(32, 32).to(self.device)
        self.tail = nn.Linear(32, 10).to(self.device+1)

    def forward(self, x):
        x = self.head(x).to(self.device+1)
        x = self.tail(x).to(self.device)
        return x


class ToyData(Dataset):
    def __init__(self):
        self.data = [[float(x)]*32 for x in range(1 << 20)]
        self.y = [x % 10 for x in range(1 << 20)]

    def __getitem__(self, idx):
        return np.asarray(self.data[idx]), self.y[idx]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--groups_size', default=2, type=int,
                        help="num. of GPUs the model take")
    parser.add_argument('--group_per_node', default=4, type=int,
                        help="num. of model replicas a node can accomondate")
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    args = parser.parse_args()

    # initialization
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    gs, gpn, rk = args.groups_size, args.group_per_node, dist.get_rank()
    start_dev_id = gs*(rk % gpn)
    torch.cuda.set_device(start_dev_id)

    # initialize model, dataloader, and train (see next section)

    is_dist = True
    bsz = args.batch_size
    model = ToyModel()
    dataset = ToyData()
    if is_dist:
        model = DDP(model)
        sampler = DistributedSampler(dataset)  # will shuffle by default
        data_loader = DataLoader(
            dataset, batch_size=bsz, shuffle=False, num_workers=2, pin_memory=True, sampler=sampler)
    else:
        # compare with regular data loader
        data_loader = DataLoader(
            dataset, batch_size=bsz, shuffle=True, num_workers=2, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss = torch.nn.CrossEntropyLoss()
    for idx, (x, y) in enumerate(data_loader):
        optimizer.zero_grad()
        x, y = x.float().cuda(), y.cuda()
        o = model(x)
        l = loss(o, y)
        l.backward()
        optimizer.step()
        if dist.get_rank() == 0:
            print(idx, (idx+1)*bsz*dist.get_world_size(), l.item())

    dist.destroy_process_group()
