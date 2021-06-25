# Structure From Motion

![Project Image](https://raw.githubusercontent.com/patel-nisarg/Structure_from_motion/main/images/samples/fountain_pt_cloud_compare.PNG)

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

*To learn more about the theory and mathematics behind SFM, **see my documentation [here](https://github.com/patel-nisarg/Structure_from_motion/blob/main/docs/Documentation.pdf).***

[Back To The Top](#read-me-template)

---

## How To Use

Program can be run from CLI as shown in the example below. A text file containing image paths should be specified for first input and the calibration matrix for the camera as the second input. To calibrate your own camera for your own set of images, you may use [camera_calibration.py](https://github.com/patel-nisarg/Structure_from_motion/blob/main/camera_calibration/camera_calibration.py) and follow the instructions [here](https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html).


### Requirements
Python 3.6.x

For package requirements, see [requirements.txt](https://github.com/patel-nisarg/Structure_from_motion/blob/main/requirements.txt).

### CLI Example

The following command runs the entire process and produces a point cloud visualization of images contained in "sfm_image_paths.txt" for a calibration matrix "K.txt". You can use these text files in the repo to get an idea of how to structure them. Using the camera_calibration.py will automatically structure the calibration matrix "K" to required dimensions.

```powershell
(venv) C:\SFM>python main.py -i "sfm_image_paths.txt" -K "K.txt"
```


[Back To The Top](#read-me-template)

### Samples

Images for a fountain are provided in the /images/datasets/fountain directory. These images produce the following point cloud.

![ptcloud_fountain](https://github.com/patel-nisarg/Structure_from_motion/blob/main/images/samples/fountain_pt_cloud2.PNG)
https://user-images.githubusercontent.com/74885742/123357787-36404a80-d538-11eb-8179-0e1c98924cb7.mp4
The [temple ring dataset](https://vision.middlebury.edu/mview/data/) can also be used and produces the following point cloud. 

![ptcloud_fountain](project-image-url)

---

## References
- [SFM datasets for benchmarking](https://github.com/openMVG/SfM_quality_evaluation)

- [Temple ring dataset](https://vision.middlebury.edu/mview/data/)

- [MATLAB SfM data structures](https://www.mathworks.com/help/vision/structure-from-motion-and-visual-slam.html)

- [Bundle adjustment using SciPy](https://github.com/patel-nisarg/Structure_from_motion/blob/main/camera_calibration/camera_calibration.py)

- [Multiple View Geometry Theory](https://books.google.ca/books/about/Multiple_View_Geometry_in_Computer_Visio.html?id=si3R3Pfa98QC&source=kp_book_description&redir_esc=y)

- [Learned correspondences code/paper](https://github.com/vcg-uvic/learned-correspondence-release)

[Back To The Top](#read-me-template)

---

## License

[MIT License](https://github.com/patel-nisarg/Structure_from_motion/blob/main/LICENSE)



[Back To The Top](#read-me-template)

---

## Author Info


- LinkedIn - [Nisarg Patel](www.linkedin.com/in/nisarg-patel-52202a158)

[Back To The Top](#read-me-template)
