# Model Report

## ENVIROMENT AND SET-UP

This project start with the proposal of training a model from a dataset of pictures in order to identify the model of the car/truck depicted on the picture. The pictures correspond to 196 different classes correctly labeled.\
The minimun accuracy demanded from the model on the test dataset is 30%.\

The models were trained on an external server with a GPU.\
Data from GPU:
- NVIDIA-SMI 470.129.06
- Driver Version: 470.129.06
- CUDA Version: 11.4
- 11441MiB

The training of the model lasted up to 6 hours in some cases when training was done with 100 epochs. Yet in less than 2 hours it has been seen that good results can be met.

## TESTS

<br>

### Initial Tests (exp_001, exp_002)

<br>

Initial test were done on the available images without any modification on them prior to the model.\
Main caracteristics:
- Batch size: 50 images.
- Dropout rate: 0.4
- Learning rate: 0.0001
- Random flip mode: "horizontal_and_vertical"
- Random rotation factor: ~0.25
- Random zoom heightfactor: ~0.25
- Random zoom width_factor: ~0.25\
\
Validation accuracy ~60%

### Crop images (exp_003)

<br>

From experiment number three onwards the models were trained on cropped images from the available ones.\
The crop of the images was done by selecting the bigger bounding box of entities selected as "car" or "truck" by Detectron2.\
Main caracteristics:
- Batch size: 64 images. (constant till the end)
- Dropout rate: 0.4
- Learning rate: 0.0001 (constant till the end)
- Random flip mode: "horizontal_and_vertical"
- Random rotation factor: 0.25
- Random zoom heightfactor: 0.25
- Random zoom width_factor: 0.25\
\
Validation accuracy ~75%\

![exp_003](/report/exp_003_accuracy.JPG)
![exp_003](/report/exp_003_loss.JPG)

### Initial regularization (exp_004, exp_005)

<br>

L1 and L2 were added as regularizers with default values (0.001)
Main caracteristics:
- Dropout rate: 0.4 -> 0.5
- Random flip mode: "horizontal_and_vertical"
- Random rotation factor: 0.35
- Random zoom heightfactor: 0.25
- Random zoom width_factor: 0.25\
\
Validation accuracy ~75%\

![exp_004/5](/report/exp_004_a_005_accuracy.JPG)
![exp_004/5](/report/exp_004_a_005_loss.JPG)

### Fine tunnig / dealing with overfitting (exp_006, exp_007, exp_008, exp_009)

<br>

At this point there was a search to reduce the breach between train and validation through regularization.\
L1 and L2 were incremented, the drop out was tested at 0.45 and 0.5.\
Since the model expected the photos of car not being upside-down 'random flip mode' was set to horizontal.\
A new regularizer layer was added at exp_008: random contrast. 
Main caracteristics:
- Adding L1_L2 regularizer. Values from 0.002 up to 0.08
- Dropout rate: 0.45 -> 0.5
- Random flip mode: "horizontal"
- Random rotation factor: 0.35
- Random zoom heightfactor: 0.25
- Random zoom width_factor: 0.25
- Random contrast factor: 0.20\
\
On most trains the behavior of loss and accuracy was similar.\
Validation accuracy ~80%\

![exp_006/9](/report/exp_006_a_009_accuracy.JPG)
![exp_006/9](/report/exp_006_a_009_loss.JPG)

### The last battle (exp_010)

<br>

At this point the experiment 008 was selected as the best model with 82% validaton accuracy and the performance on test was evaluated.\
Accuracy in test fell to 1.8%, what triggered a debugging process.\
What was identified was an incorrect usage of an argument on 'resnet_50.py'.\
Line 85 was changed from: "x = base_model(x, training = True)" to "x = base_model(x)".\
At this point a new model was trained with identical weights of experiment 008.\
The result on test dataset was aceptable within the boundaries of the project. Due to the lack of time no further models were trained.

![exp_010](/report/exp_010_accuracy.JPG)
![exp_010](/report/exp_010_loss.JPG)

## RESULTS

<br>

The Model Evaluation notebook reveal a 74,16% accuracy on the test images.\
Overall precition: 79\
Overall recall: 74

<br>

# Improvement possibilites
More models should be trained given the circumstances through experiments 001 to 009.\
On top of that the gap between train and validation data is still big.\
If even more improvements are meant to be done increasing the data available is a way to go. A simpler option could be reorganize the images in train and test; 80-20% rather than 50-50%.