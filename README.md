# Error Resilient Recurring Gallery Building (ERRGB)

<img src=/results/qualitative/three_static.png width="300">
<img src=/results/qualitative/two_dynamic.png width="300">

## Abstract
In person re-identification, people must to be correctly identified in images that come from different cameras or are captured at different points in time. In the open-set case, the above needs be achieved for persons that have not been previously observed.
In this paper, we propose a universal method for building a multi-shot gallery of observed reference identities recurrently online. We perform L2-norm descriptor matching for gallery retrieval using descriptors produced by a generic closed-set re-identification system. We continuously update the multi-shot gallery by replacing outlying descriptors with newly matched descriptors. Outliers are detected using the Isolation Forest algorithm. Thus, we ensure that the gallery is resilient to erroneous assignments, leading to improved re-identification results in the open-set case.

## Requirements
All required modules are contained in the [requirements.txt](requirements.txt).

## Downloading Model Weights
We produce descriptors of people using the Aligned ReID network as an underlying architecture. We use [this](https://github.com/huanghoujing/AlignedReID-Re-Production-Pytorch) implementation of the network. Before testing our approach, you need to clone this repository into this folder. The weights for the trained AlignedReID model should be downloaded via the link provided in the above repository and should be placed in the [models/alignedReID](models/alignedReID) folder. For testing our approach qualitatively on custom video footage, we further require the weights of YOLOV3 for person detection. The corresponding files should be placed in [models/yolov3](models/yolov3). We provide all the files as LFS files in this repository.

## Quantitative Analysis
We evaluated our approach quantitatively on the evaluation set of the Market1501 dataset. The test dataset is adapted for the open-set case by computing a random hold-out set.The identities associated with the descriptors contained in the hold-out set are removed from the gallery, since the correspond to unknown identities. 

There is a script [here](create_csv_dataset.py) to read the evaluation dataset of Market1501 (consisting of gallery and query set) into a .csv file, which is required for processing the data for evaluation. In [dataset.py](dataset.py), you will find the Dataloader, a function for splitting the evaluation query set into known and unknown identities, as well as the preprocessing done on images before forwarding them to the AlignedReID model. 

If you want to redo the evaluation, run

```
python3 run_quantitative.py
```

All parameters related to quantitative evaluation are located in [config.py](config.py)
### Results

\# Unknowns (Perc. total query set) | rank1 | TTR | FTR | 
--- | --- | --- | --- |
100 (13.3%) | 84.8% | 83.5% | 9.1% | 
375 (50%)| 39.6% | 84.4% | 6.2% |
500 (66.7%)| 91.7% | 85.7% | 5.9% |


## Qualitative Analysis
The method can also be tested on custom video data. Simply run

```
python3 run_qualitative.py --input "PATH_TO_VIDEO" --output "PATH_TO_DEST"
```

PATH_TO_VIDEO determines the relative path to the video which should be processed. PATH_TO_DEST is the relative path where the processed video should be stored.
Video data can be recorded through

```
python3 run_webcam_rotated.py
```
This will access your webcam and write the video data to the [data](data) directory. Note that the video will be rotated by 180°, which was necessary in our particular camera setup.

Please note that the code was produced for running on Google Colab. Video codecs might need to be changed depending on the underlying operating system.

## Results
Two videos from the qualitative analysis can be viewed in the [results](results) folder.
