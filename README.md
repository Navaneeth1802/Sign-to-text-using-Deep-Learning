<h1>Goal:Create 2 Models using Deep learning which can Convert Sign Language to text</h1>
<p>This project focuses on developing two independent sign-to-text translation models using deep 
learning techniques. The first model will be fingerspelling that utilize a Convolutional Neural 
Network (CNN) to recognize individual signs from the American Sign Language (ASL) 
alphabet. The second model will help us to translate non-manual features and will use a Long 
Short-Term Memory (LSTM) network to translate dynamic gestures like "hello," "thanks," 
"food," and "help" into text</p>

<h3 align="left">Languages and Tools:</h3>
<p align="left"> 
    <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQWdQwjIMDLuBRQ8zLh1NDGpsraQ1uMVSNqOQ&s" alt="c" width="60" height="60"/> </a> 
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="60" height="60"/></a>
    <img src="https://images.javatpoint.com/tutorial/keras/images/keras.png" alt="java" width="60" height="60"/> </a> 
    <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="60" height="60"/>
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1200px-Scikit_learn_logo_small.svg.png" alt="tensorflow" width="60" height="60"/>
    <img src="https://www.freecodecamp.org/news/content/images/size/w2000/2020/09/numpy-1.png" alt="tensorflow" width="60" height="60"/>
    <img src="https://miro.medium.com/v2/resize:fit:2000/1*S8Il5ethl3YFh0M9XKVz-A.png" alt="tensorflow" width="100" height="80"/>
    <img src="https://miro.medium.com/v2/resize:fit:1120/1*Hgg6bLceoIjubE2hBiJK4g.png" alt="tensorflow" width="100" height="60"/>
</p>

<h4>Model 1: ASL Alphabet Recognition</h4>

- This model utilizes a Convolutional Neural Network 
(CNN) to recognize individual ASL hand signs from images or video frames.
- We used the OpenCV (cv2) and operating system (os) libraries primarily for creating the dataset.Each letter is stored in a specific file named after the letter itself. We collected nearly 
600 images for each alphabet
- Our Model had 3 Convalutional Layer with 64,128,256 number of filters and 3 fully Connected layer with decreasing numbers of neurons (512, 256, 64) to 
progressively combine the extracted features from the convolutional layers and ReLU
activation.
- Realtime detection:Webcam stream is captured using OpenCV's video capture class and in that The ROI region is captured which is converted to grayscale and resized to 48X48 to fit the model 
requirements. The pre-processed image is then fed into loaded CNN model which predicts the 
probabilities of each possible sign language(A-Z), the predicted character with 
highest probability is defined using argmax function and its corresponding confidence score is 
displayed in the output frame.

<img src="https://github.com/user-attachments/assets/6330cebe-e7c2-48f4-a577-9148f0390a0c"  height=250 width=750>

- The model achieved a final training accuracy of 98.40%,This means that on the training set, the model correctly classified 98.40% of the 
sign language gestures. We have used the Categorical-cross entropy as the loss function the
training loss of was almost 0.0554 average across all samples in batch or epoch.The model's performance on unseen 
data, evaluated using the validation set, is even more promising as it achieved a validation 
accuracy of 99%.

<h4>Model 2: Gesture Recognition</h4> 

- This model employs a Long Short-Term Memory (LSTM) 
network in conjunction with MediaPipe for hand pose estimation to translate dynamic gestures 
into text
- We use a combination of media pipe and lstm for gesture classification, we import holistic 
module from mediapipe library this helps us to detect human pose landmarks, face landmarks 
and hand landmarks simultaneously.
- The mediapipe creates a NumPy array containing the x, y, z coordinates, and visibility 
information If there is no extracted data, then it creates a NumPy array of size then a NumPy 
array is created of size 33*4 () similar logic is applied for extracting key points of face, right 
hand and left hand and using this Array we are training out Model.
- This NumPy arrays (individual frame key points) for each sequence is transformed into a single 
sequence element within the sequences list. Each sequence element is a list containing the 
KeyPoint data for all frames within that sequence
- The the first layer of model has input shape of 30(1 sequence) each element is a vector of 
1662 dimensions. The first layer has 64 units which aids the complexity of the layer and its 
capacity to learn patterns within the data while the second layer has 128 and final LSTM layer has 64 units.
- Fully Connected Layers
  - The the first fully connected (dense) layer has 64 units. Dense layers transform the learned 
representation from the final LSTM layer into a format suitable for classification. The relu
activation function is used for introducing non-linearity within the dense layer.
  - Dense Layer-2 This is has 32 units. It further processes the output from the previous 
dense layer, potentially refining the representation for classification.
  - Dense Layer-3 is the final output layer. The number of units in this layer is set to the number of sign 
language actions in our dataset.

- Real Time Prediction
   - We will set an empty list to store the sequence, as the webcam continuously captures once the sequence 
reaches 30 frames the model will analyse it to predict the most probable sign action that we 
have performed. Along with that we use coloured bar graphs to represent model’s confidence 
level of different signs on the screen
<img src="https://github.com/user-attachments/assets/fa7aacbf-c750-4d5d-aabc-d7740813dbdc">

  - The LSTM model achieved a final training accuracy of 95.87%, indicating a good fit on the 
training data,We have used the Categorical-cross entropy as the loss function the 
training loss of was 0.1260 average across all epochs suggesting model learned the patterns in 
the training data effectively. The model's performance on unseen data that is test data, achieved
accuracy of 95%, whereas the loss was 0.2704

