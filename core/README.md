
## FedScale Automated Runtime: Evaluation Platform for Federated Learning

Existing FL evaluation platforms can hardly reproduce the scale of practical FL deployments and often fall short in providing user-friendly APIs, 
thus requiring great developer efforts to deploy new plugins. As such, we introduce FedScale Automated Runtime (FAR), 
an automated and easily-deployable evaluation platform, to simplify and standardize the FL experimental setup and model evaluation under a practical setting. 
FAR is based on [Oort project](https://github.com/SymbioticLab/Oort), which has been shown to scale well and can emulate FL training of thousands of clients 
in each round.

## Preliminary

The training evaluations rely on a distributed setting of GPUs/CPUs via the Parameter-Server (PS) architecture. 
We typically, run each experiment on 4 GPUs to simulate the FL aggregation of 10 or 50 participants in each round. 
Each training experiment is pretty time-consuming, as each GPU has to run multiple clients (10/4 or 50/4 depending on the setting) for each round. 

The following are some reference numbers on Tesla P100 GPUs for each line in our plots when using 100 participants/round for reference as detained in [here][https://github.com/SymbioticLab/FedScale/tree/master/fedscale/core].
(we also provide estimated prices on [Google Cloud](https://cloud.google.com/products/calculator), but they may be inaccurate): 

| Setting      | Time to Target Accuracy  | Time to Converge |
| ----------- | ----------- | ----------- |
| YoGi             | 53  GPU hours (~$97)     |    121  GPU hours (~$230) |

Table 1: GPU hours on Openimage dataset with ShuffleNet

### Setting Job Configuration

We provide an example of submitting a single training job in ```REFL/core/evals/manager.py```, whereby the user can submit jobs on the master node. 

- ```python manager.py submit [conf.yml]``` will submit a job with parameters specified in conf.yml on both the PS and worker nodes. 
We provide some example ```conf.yml``` in ```FedREFLScale/core/evals/configs``` for each dataset. 
They are close to the settings used in our evaluations. Comments in our example will help you quickly understand how to specify these parameters. 

- ```python manager.py stop [job_name]``` will terminate the running ```job_name``` (specified in yml) on the used nodes. 


***all logs will be dumped to ```log_path``` (specified in the config file) on each node. 
```training_perf``` locates at the master node under this path, and the user can load it with ```pickle``` to check the time-to-accuracy performance. 
Meanwhile, the user can check ```/evals/[job_name]_logging``` to see whether the job is moving on.***

## Repo Structure
```
Repo Root
|---- evals     # Backend of job submission including the manager.py and config files in configs folder
|---- utils     # Utiliy and helper modules such as dataloaders, decoder, data divider, models, etc
|---- helper    # client object and its configurations based on the device and behaviour trace file
|---- testlibs  # scripts to test for the various python modules
```
## Key Files
```
aggregator.py: this represents the FL server aggregator (can run on a GPU or CPU)
executor.py: this represents the worker that runs and executes the training for clients (runs on GPU)
resource_manager.py: assigns resources (or clients) to the executors
client_manager.py: responsible for the selection of the clients
client.py: represents the client object and the training functionality
argparser.py: contains all the arguments of related to the experiments
```

