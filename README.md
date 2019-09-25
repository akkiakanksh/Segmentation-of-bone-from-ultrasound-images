This repository consists of code used in project for "Segmentation of bone from ultrasound images using deep learning". In this project, the main programming language used is Python 
1. Dataset contains of 600 images which are divided into train set(500 images) and test set(100 images)
2. To check the bone area in an image, line profiles are drawn in Matlab to see the shadow effect
3. Masks are generated for train set in Matlab using imfreehand function
4. U-Net convolutional network is used for Biomedical image segmentation. This U-Net model is used to segment bone in ultrasound image
5. Images are resized to 256 x 256
6. Dataset and masks generated are passed to U-Net model. To get the desired output model is trained
7. Output obtained are stored in a particular folder for futher usage 
