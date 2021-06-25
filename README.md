# Structure From Motion

![Project Image](project-image-url)

This project covers the [structure from motion](https://en.wikipedia.org/wiki/Structure_from_motion) (SFM) technique for estimating three-dimensional structure from a sequence of two-dimensional images. SFM photogrammetry provides a non-invasive low-cost method for 3D reconstruction.

---

### Table of Contents

- [Structure From Motion](#structure-from-motion)
    - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [How To Use](#how-to-use)
    - [Requirements](#requirements)
    - [CLI Example](#cli-example)
    - [Samples](#samples)
  - [References](#references)
  - [License](#license)
  - [Author Info](#author-info)

---

## Description

The program consists of three phases: feature extraction, point cloud generation, and visualization.

Feature extraction can be done from this repository here which extracts features using SIFT or it can be done using [this](https://github.com/patel-nisarg/learned-correspondence-release/blob/master/generate_matches.py) learned correspondences repository which performs feature extraction automatically. This repository uses SIFT for feature extraction and a neural network for filtering features between image pairs. Once the features for each image pair are avaialable, they can be used in the main program as is.

Point cloud generation is performed by first creating a baseline of two images using these correspondences and triangulating points. More images are added subsequently to the point cloud and bundle adjustment is done at each integration of image.

Visualization of point cloud is performed using PyVista once all points are generated and bundle adjustment is complete.

To read more about the theory and mathematics behind SFM, see my documentation [here]().

[Back To The Top](#read-me-template)

---

## How To Use

Program can be run from CLI as shown below. A text file containing image paths should be specified for first input and the calibration matrix for the camera as the second input. To calibrate your own camera for your own set of images, you may use camera_calibration.py and follow the instructions [here](https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html).


### Requirements
See requirements.txt

### CLI Example

The following command runs the entire process and produces a point cloud visualization of images contained in "sfm_image_paths.txt" for a calibration matrix "K.txt". You can use these text files in the repo to get an idea of how to structure them. Using the camera_calibration will automatically structure the calibration matrix "K" to required dimensions.

```python
(venv) C:\SFM>python main.py -i "sfm_image_paths.txt" -K "K.txt"
```


[Back To The Top](#read-me-template)

### Samples

Images for a fountain are provided in the /images/fountain directory. These images produce the following point cloud.

![ptcloud_fountain](project-image-url)

The [temple ring dataset](https://vision.middlebury.edu/mview/data/) can also be used and produces the following point cloud. 

![ptcloud_fountain](project-image-url)

---

## References
[SFM datasets for benchmarking](https://github.com/openMVG/SfM_quality_evaluation)

[Temple ring dataset](https://vision.middlebury.edu/mview/data/)

[MATLAB SfM](https://www.mathworks.com/help/vision/structure-from-motion-and-visual-slam.html)

[Back To The Top](#read-me-template)

---

## License

[MIT License](https://github.com/patel-nisarg/Structure_from_motion/blob/main/LICENSE)



[Back To The Top](#read-me-template)

---

## Author Info


- LinkedIn - [Nisarg Patel](www.linkedin.com/in/nisarg-patel-52202a158)

[Back To The Top](#read-me-template)
