# Table of Contents



# Introduction:

The project is about analyzing medical images to help diagnose breast cancer. A dataset of X-rays for breast cancer images is used to identify lumps or abnormalities that may indicate cancer. Image processing techniques, such as feature extraction using Histogram of Oriented Gradients (HOG) and other techniques, are applied to analyze the shape and structure of the lumps. These features are then used to train classification models that can differentiate between benign and malignant cases. The goal is to improve diagnostic accuracy and aid in early detection of breast cancer.

## Preprocessing

This step have a key role on the performance of Machine Learning model,
* We resize the images first to make process on images faster (before size = 640 * 640 , after = 224 * 224)
* Normalization: Pixel values are normalized to a [0, 1] range to enhance model performance.
* Contrast Enhancement: The contrast is slightly increased to improve visual quality.
* The processed images are saved in a designated output folder created automatically if it doesn't already exist.


# **Feature Extraction**

**1.1- Gabor Feature Extraction:**

Gabor is a linear filter widely used in tasks like edge detection, feature extraction, and texture classification in machine learning. As a bandpass filter, it passes specific frequencies while attenuating others, making it highly effective for extracting meaningful patterns in data.
For nearly three decades, Gabor filters have been a cornerstone in computer vision and image analysis, particularly in feature extraction. Their design mimics the receptive fields of simple cells in the visual cortex, which initially drew attention. More importantly, Gabor filters have consistently excelled in applications such as face detection, iris recognition, and fingerprint matching, where they rank among the top performers.
What sets Gabor features apart is their elegant derivation across both spatial and frequency domains, grounded in the principles of signal processing. With their practical advantages and computational efficiency, Gabor filters are likely to continue playing a key role in future applications.

**1.2- Scale Invariant Feature Transform** 


The Scale Invariant Feature Transform (SIFT) detects and describes local features in images. It is robust to scale, rotation, and lighting changes. Key aspects include:

1. Keypoint Detection: Identifies stable keypoints.
2. Scale Space Extrema: Uses Gaussian blurring to find keypoints across different scales.
3. Orientation Assignment: Ensures rotation invariance by assigning a dominant orientation to keypoints.
4. Descriptor Generation: Creates feature vectors from local image gradients.
5. Feature Matching: Compares descriptors for tasks like object recognition and image stitching.

SIFT is widely used for image analysis and matching tasks.


**1.3- Local Binary Pattern (LBP):**

considers the area around each pixel and after thresholding,
uses the result as a binary integer to identify pixels in a picture.
Local Binary Pattern (LBP)The   area   around   each   pixel   is   considered   and   after thresholding  it  and  using  the  result  as  a  binary  integer,  the Local  Binary  Pattern  [1,9,11,12,13]  a  straightforward  yet very  effective  texture  operator,  identifies  the  pixels  in  a picture. LBP texture operator has been a well-liked method in many    applications    because    it    avoids    computational complexity  and  also  it  has  discriminative  capability.  It  is  a technique that unifies the statistical and structural models of texture  analysis,  which  have  often  been  different.  The  LBP operator's  resistance  to  monotonic  gray-scale  shifts  brought on  by,  say,  changes  in  lighting  may  be  its  most  crucial characteristic  in  practical  applications.  Its  computational simplicity  is  another  significant  characteristic  that  enables picture analysis in demanding real-time environments


**1.4- HOG (Histogram of Oriented Gradients):** 

In  image  processing,  HOG  feature  descriptor  is  mostly utilized  for  detection  of  objects.  A  feature  descriptor  is  a representation of an image that simplify the image by drawing out pertinent information.
To describe the appearance and shape of local objects inside an  image,  the  distribution  of  intensity  gradients  or  edge directions  can  be  used  according  to  the  theory  behind  the histogram  of  oriented  gradients descriptor.  This descriptor uses the histograms of gradient direction as features.


2- **Pretrained Models for Feature Extraction:**

### Advantages and Limitations of Using Pretrained Models for Feature Extraction

#### Advantages:
1. **High Accuracy**: Pretrained models like VGG, ResNet, and Inception have been trained on large datasets (e.g., ImageNet), enabling them to achieve high accuracy in recognizing complex features in images.
  
2. **Time Efficiency**: Using pretrained models allows you to leverage existing knowledge without the need to train a model from scratch, which can be time-consuming and resource-intensive.
  
3. **Robust Feature Extraction**: These models are capable of extracting meaningful features that can enhance the performance of downstream tasks, such as classification and object detection.
  
4. **Transfer Learning**: Pretrained models can be fine-tuned on smaller, task-specific datasets, improving their adaptability and performance for specific applications.

#### Limitations:
1. **Size and Complexity**: Pretrained models are often large and computationally demanding, requiring significant memory and processing power, which may not be feasible for all environments.
  
2. **Lack of Customization**: The features extracted may not be optimally tailored for specific datasets or applications, especially if there is a significant difference between the pretrained data and the target data.
  
3. **Dependence on Preprocessing**: These models require specific preprocessing steps (e.g., normalization, resizing) to be applied to the input data, which can introduce additional complexity and potential errors if not handled correctly.
  
4. **Overfitting Risk**: Fine-tuning a pretrained model on a small dataset can lead to overfitting, where the model performs well on training data but poorly on unseen data.

This discussion helps to understand both the strengths and challenges associated with using pretrained models for feature extraction in image processing tasks.


#  Ensemble Techniques

This script performs feature extraction from a dataset of images 
using four techniques: Gabor filters, SIFT, LBP, 
and HOG. Each method captures different aspects of the images, enabling 
 a comprehensive representation.

After extraction, the features are combined into a single feature matrix 
 This model integrates to enhance predictive accuracy by leveraging their strengths.


 # **Image Enhancement**

 **1. Histogram Equalaization (HE)**

Implementing HE to enhance image contrast is very important for ML models, since it rearrarnge the bins of intensity by transforms gray level values to the entire range and enhance the distribution.
if the image is dark, low contrast or bright -> it will lead to high contrast image ( uniformly distributed histogram) .

**2. noise rudiction**

This function applies Gaussian blur to reduce noise in images. Gaussian blur is an effective image processing technique that smooths images by averaging pixel values within a defined neighborhood around each pixel, with more weight given to pixels closer to the kernel's center. Its primary purpose is to reduce high-frequency noise and detail, enhancing subsequent image processing tasks like edge detection and segmentation. The effectiveness of Gaussian blur is influenced by the kernel size, which determines the area of pixels involved in the blurring, and the standard deviation, which controls the blurring intensity; larger kernels and higher standard deviations produce a more pronounced blur, while smaller ones maintain more image detail.

**3.  sharpening filters:**

implemented sharpening filters to enhance image edges and bring out subtle variations in contrast. This step is particularly important for detecting fine details, such as fractures in medical images. By improving the clarity of critical structures, sharpening not only enhances the overall image quality but also aids in more effective feature extraction. This ultimately contributes to higher diagnostic accuracy, as well as increased sensitivity and specificity in image analysis tasks.

**4. Additional Contrast Enhancement
Applied  Contrast Limited Adaptive Histogram Equalization (CLAHE) to boost contrast by adjusting local regions, enhancing details without amplifying noise.


## Implementation of Machine Learning Models

After reviewing various studies on X-ray classification, we found that Support Vector Machine (SVM), Random Forest (RF), and Neural Network (NN) classifiers have shown great results in accurately classifying X-ray images.

**1. Support Vectore Machine (SVM)**

is a versatile machine learning algorithm designed for tasks like classification, regression, and outlier detection. It excels at both linear and nonlinear classification, making it a popular choice across diverse domains. From text and image classification to spam filtering, handwriting recognition, gene analysis, face detection, and anomaly detection, SVMs are widely applicable. The strength of SVMs lies in their ability to find the optimal separating hyperplane between classes, which ensures accurate classification, whether it's binary or multiclass. This outline will delve into how SVMs work, their key applications, and their ability to handle a variety of tasks, including regression and detecting outliers.


**2. Random Forest (RF)**

Random Forest algorithm is a powerful tree learning technique in Machine Learning. It works by creating a number of Decision Trees during the training phase. Each tree is constructed using a random subset of the data set to measure a random subset of features in each partition. Train a Random Forest model on image data for classification of 1 and 0 images.


**3. Neural Network (NN)**

Neural Networks are computational models inspired by the structure of the human brain. They consist of layers of interconnected nodes, or neurons, that process and learn from data, making them powerful for tasks like pattern recognition and decision making in machine learning. This outline will dive deeper into how neural networks work, their architecture, and their role in various applications.



