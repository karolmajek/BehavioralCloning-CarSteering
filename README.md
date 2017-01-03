# BehavioralCloning-CarSteering
Behavioral cloning project for Udacity Self-Driving Nanodegree. Neural net that drives the car.


## Input data

Data are exported from 3 simulated cameras: left, center and right. Size of single image from one each camera is 320x160.

Example of input data:
![Input data](images/input_data.png)

### Data structure

I have created directory named *recordings*. Inside this directory I create directory for each training set. Directories are named with the track number and trial number.
```
recordings/
├── track0_trail0
├── track0_trail1
```
Inside of directory with trial recordings there are *driving_log.csv* with image names, steering angle, throttle, break and speed.
```
track0_trail0
├── driving_log.csv
└── IMG
    ├── center_2016_12_30_12_45_27_062.jpg
    ├── center_2016_12_30_12_45_27_181.jpg
    ├── center_2016_12_30_12_45_27_295.jpg
    ...
    ├── left_2016_12_30_12_45_27_062.jpg
    ├── left_2016_12_30_12_45_27_181.jpg
    ├── left_2016_12_30_12_45_27_295.jpg
    ...
    ├── right_2016_12_30_12_45_27_062.jpg
    ├── right_2016_12_30_12_45_27_181.jpg
    ├── right_2016_12_30_12_45_27_295.jpg
    ...
```
First 3 lines of *driving_log.csv*: (There are actually global paths...)
```
track0_trail0/IMG/center_2016_12_30_12_45_27_062.jpg, track0_trail0/IMG/left_2016_12_30_12_45_27_062.jpg, track0_trail0/IMG/right_2016_12_30_12_45_27_062.jpg, 0, 0, 0, 8.181675E-05
track0_trail0/IMG/center_2016_12_30_12_45_27_181.jpg, track0_trail0/IMG/left_2016_12_30_12_45_27_181.jpg, track0_trail0/IMG/right_2016_12_30_12_45_27_181.jpg, 0, 0, 0, 8.170124E-05
track0_trail0/IMG/center_2016_12_30_12_45_27_295.jpg, track0_trail0/IMG/left_2016_12_30_12_45_27_295.jpg, track0_trail0/IMG/right_2016_12_30_12_45_27_295.jpg, 0, 0, 0, 8.189175E-05
```
Here are names for columns (not provided in *driving_log.csv*)
```
Center image, Left image, Right image, Steering Angle, Throttle, Break, Speed
```

## Usage examples

### Visualize input data while training

```
./model.py --show_input 1
```

### Change train/val dataset csv file

```
./model.py --training_csv TRAIN_csv --validation_csv VAL_csv
```

### Number of epochs and batch Size
```
./model.py --epochs 50 --batch_size 128
```
