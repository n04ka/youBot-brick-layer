# Brick laying robot based on KUKA youBot

The following code is a modular structure of robot's control system. It includes a list of modules:
- Path planning
- Manipulator & platfrom control
- Lidar vision & mapping
- RGBD-camera vision
- Brick orientation detection
- Construction scheme input
- Strategic level

![brick-layer](https://github.com/user-attachments/assets/7498ee2d-2ed3-4680-a41e-a99a8ee01811)

## Path planning

Path planning is based on RRT* algorithm in 2D space.

![RRT*-path-planning](https://github.com/user-attachments/assets/db19b074-f4f7-4406-a718-26f4fd8ef43b)

## Manipulator & platfrom control

Both modules contain neccessary maths implemeted according to robot parameters. They include inverse kinematics and basic maneuvers.

## Lidar vision & mapping

The purpose of the module is mapping the environment for future path planning. 

![scene](https://github.com/user-attachments/assets/e331b0e5-a924-4a35-97ae-b7cfcc314df6)
![map](https://github.com/user-attachments/assets/fd43fa16-06da-4de7-a686-d78804c05a49)

## RGBD-camera vision

The main goal of the vision is to find target objects such as bricks. To achieve this yolov8 was used. You can check full dataset and the model on [my Roboflow project page](https://universe.roboflow.com/bricks-fbqb5/bricks-2). The module outputs point clouds to be analyzed by the next one.

![segmentation](https://github.com/user-attachments/assets/a8307d51-58d9-4e79-933e-c30104ddf63c)
![gif](https://github.com/n04ka/youBot-brick-layer/blob/main/%D0%B2%D0%B8%D0%B4%D0%B5%D0%BE%20%D1%81%20%D0%BA%D0%B0%D0%BC%D0%B5%D1%80%D1%8B.gif)

## Brick orientation detection

The following module analyzes point clouds and retrieves object position and orientation. Based on RANSAC algorithm it finds up to 3 planes and calculates the Euler angles that then can be used for manipulator input.

![cloud](https://github.com/user-attachments/assets/1412d667-c44f-4dde-90e9-48117f1b3885)

## Construction scheme input

The module enables user input to determine construction structure and order. The robot will build the walls according to the JSON scheme searching for materials on a construction site.

## Strategic level

Strategic lavel is based on state machine. The full brick-laying cycle is implemented and includes some emergency situations handlers:

![state-graph](https://github.com/user-attachments/assets/822a4c61-f3ac-4f82-88f2-2f3eab6b2fe9)

