
## Datasets and Benchmarks of REFL

## Getting Started

To download the dataset, run (or download from [here](https://drive.google.com/drive/folders/12s44-VmbLozTsU9oM4RGzXpBfgoeFqmy?usp=sharing) manually):

```
# Download all datasets from DropBox. Check download.sh -h for more details
bash download.sh -option 
```

## Realistic FL Datasets

***We are adding more datasets! Please feel free to contribute.***

We provide real-world datasets for the federated learning community, and plan to release much more soon! Each is associated with its training, validation and testing dataset. A summary of statistics for training datasets can be found in Table, and you can refer to each folder for more details. Due to the super large scale of datasets, we are uploading these data and carefully validating their implementations to FAR. So we are actively making each dataset available for FAR experiments. 

CV tasks:

| Dataset   | Data Type   | # of Clients | # of Samples | Example Task | 
|-----------| ----------- |--------------|--------------|    ----------- |
| CIFAR10   |   Image     | Custom       | 60K          |   Classification  |    
| OpenImage |   Image     | 13,771       | 1.3M         |   Classification, Object detection      |

NLP tasks:

| Dataset       | Data Type   |# of Clients  | # of Samples   | Example Task | 
| -----------   | ----------- | -----------  |  ----------- |   ----------- |
| Stackoverflow |   Text      |   342,477    |   135M        |  Word prediction, classification |
| Reddit        |   Text      |  1,660,820   |   351M        |  Word prediction   |
|Google Speech  |   Audio     |     2,618    |   105K        |   Speech recognition |


***Note that no details were kept of any of the participants age, gender, or location, and random ids were assigned to each individual. In using these datasets, we will strictly obey to their licenses, and these datasets provided in this repo should be used for research purpose only. ***

## Repo Structure

```
Current Folder
|---- data        # Dictionary of each datasets 
|---- donwload.sh        # Download tool of each dataset
    
```

## Example Dataset

### Google Speech Commands
A speech recognition dataset with over ten thousand clips of one-second-long duration. Each clip contains one of the 35 common words (e.g., digits zero to nine, "Yes", "No", "Up", "Down") spoken by thousands of different people. 

### OpenImage. 
[OpenImage](https://storage.googleapis.com/openimages/web/index.html) is a vision dataset collected from Flickr, an image and video hosting service. It contains a total of 16M bounding boxes for 600 object classes (e.g., Microwave oven). We clean up the dataset according to the provided indices of clients. 


### Reddit and StackOverflow
Reddit (StackOverflow) consists of comments from the Reddit (StackOverflow) website. It has been widely used for language modeling tasks, and we consider each user as a client. In our benchmark, we restrict to the 30k most frequently used words, and represent each sentence as a sequence of indices corresponding to these 30k frequently used words. We use Transformers to tokenize these sequences with a block size 64.

### Dataset of System Performance and Availability

#### Heterogeneous System Performance
This is captured in file `device_info/client_device_capacity` containing client id as key and computation and communication speed as value pairs. We use the [AIBench](http://ai-benchmark.com/ranking_deeplearning_detailed.html) dataset and [MobiPerf](https://www.measurementlab.net/tests/mobiperf/) dataset. AIBench dataset provides the computation capacity of different models across a wide range of devices. As specified in real [FL deployments](https://arxiv.org/abs/1902.01046), we focus on the capability of mobile devices that have > 2GB RAM in this benchmark. To understand the network capacity of these devices, we clean up the MobiPerf dataset, and provide the available bandwidth when they are connected with WiFi, which is preferred in FL as well. 

#### Availability of Clients
This is captured in file `device_info/client_behave_trace` as key value pair of client id, duration, finish time and a list of active and inactive time-slots of the client. We use a large-scale real-world user behavior dataset from [FLASH](https://github.com/PKU-Chengxu/FLASH). It comes from a popular input method app (IMA) that can be downloaded from Google Play, and covers 136k users and spans one week from January 31st to February 6th in 2020. This dataset includes 180 million trace items (e.g., battery charge or screen lock) and we consider user devices that are in charging to be available, as specified in real [FL deployments](https://arxiv.org/abs/1902.01046).

## Notes
please consider to cite our papers if you use the code or data in your research project.

```bibtex
@inproceedings{REFL-arxiv,
  title={Resource-Efficient Federated Learning},
  author={Ahmed M. Abdelmoniem, Atal Narayan Sahu, Marco Canini, Suhaib A. Fahmy},
  booktitle={arXiv:2111.01108},
  year={2021}
}
```

and  

```bibtex
@inproceedings{REFL-EuroSys23,
  title={REFL: Resource Efficient Federated Learning},
  author={Ahmed M. Abdelmoniem, Atal Narayan Sahu, Marco Canini, Suhaib A. Fahmy},
  booktitle={ACM EuroSys},
  year={2023}
}
```

## Contact
Ahmed M. A. Sayed, aka. Ahmed M. Abdelmoniem (ahmedcs982@gmail.com).

