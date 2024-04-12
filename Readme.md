# Text Detection using EAST Algorithm

This notebook contains code to convert a pre-trained EAST model to TensorFlow Lite (TFLite), enabling efficient deployment on various platforms. The EAST (Efficient and Accurate Scene Text Detector) pipeline excels in predicting words and lines of text at arbitrary orientations on 720p images, boasting a commendable inference speed of 13 FPS as reported by the authors.
[EAST Algorithm](eastAlgorithm.png)


## Introduction
The EAST model was introduced in the paper titled "An Efficient and Accurate Scene Text Detector". Its robustness and speed make it a popular choice for text detection tasks. This repository provides a convenient way to leverage the capabilities of EAST for text detection within your projects.


## TFLite Model Conversion

The conversion process involves exporting the pre-trained EAST model to a TFLite model. We aim to optimize inference speed and resource utilization, particularly focusing on leveraging TFLite's GPU delegate for accelerated performance.

### Float16 Model Conversion

By exporting the float16 model with a fixed known input shape, we can potentially accelerate inference using the TFLite GPU delegate. This notebook provides instructions on how to specify the `input_shapes` argument in the `tf.compat.v1.lite.TFLiteConverter.from_frozen_graph()` function to achieve this optimization.

### Integer Quantization

For further optimization, integer quantization is explored. However, this method requires a representative dataset. In this project, the [COCO-Text dataset](https://vision.cornell.edu/se3/coco-text-2/) is utilized for this purpose. A subset of 100 training images randomly sampled from the COCO-Text dataset is provided due to the dataset's size constraints.


## Performing Inference

Once the model conversion is complete, you can seamlessly perform inference using the converted TFLite model. This notebook outlines the general steps required for inference, allowing you to integrate text detection capabilities into your applications effectively.

Feel free to explore the code and adapt it to your specific use case. Contributions and feedback are always welcome!

## Demonstration!
[Reference Image](sign3.jpeg)
![Inference Image](sign3Inference.png)



