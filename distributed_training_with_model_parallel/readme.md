# Distributed Data Parallel with Model Parallel in an HPC environment

## Objective
This tutorial is on :
* 1) how to separate a model and put it on two GPUs.
* 2) how to train such model in a distributed fashion.
* 3) how to use `torch.distributed.launch` and create a slurm job script

## Model Parallel (Pipelining)
When a model is too large to fit in one GPU device, we can cut it in half and
put each part on different GPU device. To do this, we need to partition the model into 
"head" and "tail" and specify which device to put them on. In the following toy example, 
we simply put the first part in to current cuda device and the second part to the next device.

```python
import torch.nn as nn 
class ToyModel(nn.Module):
  def __init__(self):
    super(ToyModel, self).__init__()
    self.device = torch.cuda.current_device()
    self.head = nn.Linear(32,32).to(self.device)
    self.tail = nn.Linear(32,10).to(self.device+1)

  def forward(self, x):
    x = self.head(x).to(self.device+1)
    x = self.tail(x).to(self.device)
    return x
```

## Distributed Computing
Distributed computing essentially is to conduct a task using distributed and
interconnected computation resources. 
When a distributed code launches, it starts multiple processes. For training a deep learning model,
the number of (main) processes is the same as the number of model replicas.
Distributed computation requires communications amoung these processes. 
PyTorch supports various [backend communication
libraries](https://pytorch.org/docs/stable/distributed.html). 
For NVIDIA GPUs, NCCL provides the best performance due to GPUDirect. 

Assuming we have 8 GPUs on a node, and each model replica takes 2 GPUs, we can launch 4 processes 
on this node, and assign GPU0-1 to process 1, GPU2-3 to process 2 and so on. 
Here is an example code does the job.

```python
import argparse
import torch.distributed as dist

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
  
  dist.destroy_process_group()
```

## Distributed Data Parallel
PyTorch provides a very handy wrapper class `DistributedDataParallel` and a
corresponding `DistributedSampler` for loading data. Here is an example code
to show how to use these two classes together.

```python
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

class ToyData(Dataset):
    def __init__(self):
        self.data = [[float(x)]*32 for x in range(1 << 20)]
        self.y = [x % 10 for x in range(1 << 20)]

    def __getitem__(self, idx):
        return np.asarray(self.data[idx]), self.y[idx]

    def __len__(self):
        return len(self.data)


if __name=='__main__':
    # after the previous cuda setup code 
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
```

## Launching PyTorch Distributed Training
PyTorch provides a launch script `torch.distributed.launch`, where you can
configure the distributed environment.  It is usually a good idea to keep a
shell script to handle such information. See the following example. 
Note that the environment dependent variables are surrounded by angle bracket `<opt>`.

```bash
#!/bin/bash
## for torch distributed launch
## For master node
nnodes=2               # how many nodes in this computation
node_rank=0            # current node rank, 0-indexed
nproc_per_node=4       # number of models per node
master_addr=<hostname> # you should use either an ip address (i.e. AWS),
                       #   or a node name on hpc environment.
port=<port_number>     # for example, 8888

python -m torch.distributed.launch \
    --nproc_per_node ${nproc_per_node} \
    --nnodes ${nnodes} \
    --node_rank ${node_rank} \
    --master_addr ${master_addr} \
    --master_port 8888 \
    main.py \
    --groups_size 2 \
    --group_per_node ${nproc_per_node}
```

```bash
#!/bin/bash
## for torch distributed launch
## For worker node
## **all the same, except  node_rank = 1**
nnodes=2               # how many nodes in this computation
node_rank=1            # current node rank, 0-indexed
nproc_per_node=4       # number of models per node
master_addr=<hostname> # you should use either an ip address (i.e. AWS),
                       #   or a node name on hpc environment.
port=<port_number>     # for example, 8888

python -m torch.distributed.launch \
    --nproc_per_node ${nproc_per_node} \
    --nnodes ${nnodes} \
    --node_rank ${node_rank} \
    --master_addr ${master_addr} \
    --master_port 8888 \
    main.py \
    --groups_size 2 \
    --group_per_node ${nproc_per_node}
```

## Working with Slurm
To work with Slurm, let's add some arguments such as `nnodes`, `node_rank` and `master_addr`.
Let's call this bash script `launch.sh`.

```bash
#!/bin/bash
## for torch distributed launch
## **all the same, except  node_rank = 1**
nnodes=$1               # total number of nodes used in this computation
node_rank=$2            # current node rank, 0-indexed
nproc_per_node=4        # number of processes (models) per node
master_addr=$3          # hostname for the master node 
port=8888               # 

source activate <pytorch_venv> ## if using conda 
python -m torch.distributed.launch \
    --nproc_per_node ${nproc_per_node} \
    --nnodes ${nnodes} \
    --node_rank ${node_rank} \
    --master_addr ${master_addr} \
    --master_port 8888 \
    main.py \
    --groups_size 2 \
    --group_per_node ${nproc_per_node}
```

Then, let's create a Slurm job script, named `job.sbatch`, assuming we are
using 2 nodes and each node has 8 GPUs hosting 4 models, as each model
occupying 2 GPUs. The `job.sbatch` will figure out which nodes are assigned for this job, and run
the `launch.sh` on each node with correct arguments. You should change the `partition`, `account` 
and `job-name` to your use case. And also remember to load correct modules (HPC environment dependent).

```bash
#!/bin/bash

#SBATCH --partition=<partition_in_your_hpc>
#SBATCH --nodes=2
#SBATCH --time=1:00:00
#SBATCH --account=<your_account>
#SBATCH --exclusive
#SBATCH --job-name=<your_choice>

var=(`scontrol show hostname $SLURM_NODELIST`)
node_sz=${#var[@]}

# load the necessary modules, depend on your hpc env
module load anaconda3

for i in `seq 0 $(echo $node_sz -1 | bc)`;
do
    echo "launching ${i} job on ${var[i]} with master address ${var[0]}"
    srun -w ${var[$i]} -N 1 -n 1 -c 24 launch.sh ${node_sz} ${i} ${var[0]} &
done

wait
```

To summarize, the `job.sbatch` launches `launch.sh` on all the working nodes,
and `launch.sh` executes the distributed pytorch code `main.py`. 
A working example can be found
[here](https://github.com/YHRen/PyTorchDemos/tree/master/distributed_training_with_model_parallel)
