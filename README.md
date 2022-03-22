# gazebo_kinect
A package utilizes machine vision tools and alogrithms in gazebo simulation environment.


the first approach was to utilize the HSV mask using openCV API, HSV masking is quite simple and  isn't a computationally expensive algorithm 

the second approach was to utilize the mobilenet model using tensorFlow API, similar to HSV masking it's relativly fast and quite easy to use

the main goal of the implemented algorithms was object detection, both alogrithms were capable of detecting the position of the detected objects 


here HSV masking was used to detect the position of each cube according to it's color 

<img src="https://user-images.githubusercontent.com/63298005/159407160-aeb988e2-9d19-4b45-b9db-04dd3cdc1ace.png" width="250" height="250"> <img src="https://user-images.githubusercontent.com/63298005/159407377-3874e901-bac2-4772-a434-c1329d821015.jpeg" width="500" height="250">

the tensorFlow model uses mobilenet pretrained model to identify the class of the object and detect the object's position
![WhatsApp Image 2022-03-16 at 4 59 56 PM](https://user-images.githubusercontent.com/63298005/159408291-7d568748-393f-4e33-b5d9-241ea4418d48.jpeg)



## launch

start the simulation by typing the following command in the terminal `roslaunch camera camera.launch`

the implemented model is chosen in the `camera.launch` file. 
