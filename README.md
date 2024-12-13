# Gesture Recognition Robot

This repository contains the code for a robot that can recognize and respond to human gestures using a camera and machine learning.

The primary feature of our robot is its ability to recognize gestures. Our original idea for implementing this was using object segmentation to find the hand, and then another algorithm for figuring out the gestures: but this would likely use up an enormous amount of processing power. After further research, we found a better solution using Google’s MediaPipe Hand Landmarker tool combined with a custom neural network made with Keras. The hand landmarker takes in an image and produces an output of the hand’s “key points”, which are mainly the joints in each finger and the wrist of the hand. Given that this produces exactly 21 points every time it finds a hand in the image, we could create a neural network that takes in these 21 inputs and produces an output gesture based on hand training data that we could create.

We also implemented a mini GOAT (GO to AnyThing) command as a gesture. There are two steps to this process: memorization and navigation. Memorization only occurs when given a specific gesture. Upon recognition, the robot will immediately memorize eight objects within its view with the highest confidence scores, filtering out people, chairs, tables, and other common objects that are not to be remembered. It also filters out objects based on the relative depth estimates, so distant objects will not be relevant to prevent a fire extinguisher that is 20 feet away from being memorized as a bottle. After the filtering process, the remaining objects will become keys in a dictionary and assigned the robot’s pose as a value.

For the filtering process, we had to implement Non-Maximum Suppression due to the many overlapping guesses that the segmentation algorithm produced. Non-Maximum Suppression is used to output the best “bounding box” out of a set of overlapping bounding boxes; where bounding box refers to a rectangle that represents the object’s location in the frame. This is done by calculating the Intersection over Union (Area of intersection / Combined Area) of all bounding boxes and taking the highest confidence box that is above the Intersection over Union “threshold”, which determines if a box is significantly overlapping or not.



## Usage
First bring up the master node on the robot
Then do ```cd gesture/src/movebase/launch``` and ```roslaunch move_base.launch```
Finally, run ```python roscam.py```
Minimum python version 3.x

## Links
https://github.com/campusrover/GestureBot

https://campusrover.github.io/labnotebook2/faq/cv/gesture-recognition/

https://campusrover.github.io/labnotebook2/faq/ml/monocular-depth-estimation/

https://campusrover.github.io/labnotebook2/reports/2024/GestureBot/