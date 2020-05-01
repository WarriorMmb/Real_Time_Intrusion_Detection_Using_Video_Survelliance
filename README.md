# Real_Time_Intrusion_Detection_Using_Video_Survelliance
## Intro

[This is a part of my research work titled *Deep Learning based Real Time Crime Detection Using Video Survelliance](http://dspace.nitrkl.ac.in/dspace/handle/2080/3455)

## Requirements

Python3, tensorflow 1.0, numpy, opencv 3. Links for installation below:


- [Python 3.5 or 3.6, Anaconda](https://www.youtube.com/watch?v=T8wK5loXkXg)
- [Tensorflow](https://www.youtube.com/watch?v=RplXYjxgZbw&t=91s). I recommend using the tensorflow GPU version. But if you don't have GPU, just go ahead and install the CPU versoin.<br>GPUs are more than 100x faster for training and testing neural networks than a CPU. Find more [here](https://pjreddie.com/darknet/hardware-guide/)
- [Opencv](https://anaconda.org/conda-forge/opencv)
- [pygame](https://www.pygame.org)
- [bokeh](https://docs.bokeh.org/en/latest/index.html)

# step-1
## Download the Darkflow repo

- Click [this](https://github.com/thtrieu/darkflow)
- Download and extract the files somewhere locally

# Step-2
## Build the Darkflow

You can choose _one_ of the following three ways to get started with darkflow. 
1. Just build the Cython extensions in place. NOTE: If installing this way you will have to use `./flow` in the cloned darkflow directory instead of `flow` as darkflow is not installed globally.
    ```
    python3 setup.py build_ext --inplace
    ```

2. Let pip install darkflow globally in dev mode (still globally accessible, but changes to the code immediately take effect)
    ```
    pip install -e .
    ```

3. Install with pip globally
    ```
    pip install .
    ```
# Step-3
## Download a weights file

- Download the YOLOv2 608x608 weights file [here](https://pjreddie.com/darknet/yolov2/)
- Read more about YOLO (in darknet) and download weight files [here](http://pjreddie.com/darknet/yolo/). In case the weight file cannot be found, you can check [here](https://drive.google.com/drive/folders/0B1tW_VtY7onidEwyQ2FtQVplWEU), which include `yolo-full` and `yolo-tiny` of v1.0, `tiny-yolo-v1.1` of v1.1 and `yolo`, `tiny-yolo-voc` of v2. Owner of this weights is [Trieu](https://github.com/thtrieu).
- NOTE: there are other weights files you can try if you like
- create a ```bin``` folder within the ```darkflow-master``` folder
- put the weights file in the ```bin``` folder

# Step-4
## Download my repo and copy all the content inside darkflow folder

# Step-5
## Run the ```final_product.py```


# RESULT
### See the result video (```result.mp4```) in my repo. I also added my research paper in this repo. You can follow that.

# References

- Real-time object detection and classification. Paper: [version 1](https://arxiv.org/pdf/1506.02640.pdf), [version 2](https://arxiv.org/pdf/1612.08242.pdf).

- Simple Online and Realtime Tracking [paper](https://arxiv.org/pdf/1602.00763)

- Official [YOLO](https://pjreddie.com/darknet/yolo/) website.

- I have learned YOLO, how it works from [coursera](https://www.coursera.org/lecture/convolutional-neural-networks/yolo-algorithm-fF3O0). Also Siraj has a nice [tutorial](https://www.youtube.com/watch?v=4eIBisqx9_g&t=1170s) on it. 

- The original darkflow repo is [this](https://github.com/thtrieu/darkflow) by [Trieu](https://github.com/thtrieu).

<br>
