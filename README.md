# Face Mask Detector
The code used to warp images of masks onto people's faces for my dataset generateion can be found here:
[DataGenerator](https://github.com/prajnasb/observations/tree/master/mask_classifier/Data_Generator)

I created a Face Mask Detector that puts green boxes around faces that wear masks and red boxes around faces that are not wearing masks. I built this project as it can potentially be used to help ensure the safety of others.
![](maskdetector.gif)
## Dependencies
- python 3.7
- Tensorflow 2
- In addition, please `pip install -r requirements.txt`
 
## Dataset
I used roughly 2,000 images from CelebA, half wearing fake masks. The fake masks were intelligently edited on top of people’s faces.

<img src="dataset_examples.PNG" width="500"> <img src="masks.PNG" width="350">

## Running Mask Detector
1. Change to correct directory:
    ```bash
    cd code/
    ```
2. Run script
    ```bash
     python predict.py [path_to_image]
    ```
    If you do not put a path, the script will automatically turn on your webcam and predict upon those live frames.
## Results on Images
![Example 2](predicted2.PNG) ![Example 6](predicted1.PNG)  
