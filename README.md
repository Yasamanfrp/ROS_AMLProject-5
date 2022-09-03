# ROS_AMLProject-5


## About

The project is a simplified version Open-Set Domain Adaptation through Self-Supervision.


### requirements

All prerequisites can be found in [requirements.txt](requirements.txt).

## Launch

run `main.py` with the desired arguments:

- --source: source domain
- --target: target domain
- --path_dataset: path where the dataset is located
- --epochs_step1: number of epochs of step1 for known/unknown separation
- --epochs_step2: number of epochs of step2 for source-target adaptation
- --weight_RotTask_step1: weight for the rotation loss in step1
- --weight_RotTask_step2: weight for the rotation loss in step2
- --threshold: threshold for the known/unkown separation

## Acknowledgements

Thanks to Silvia Bucci for her assistance and guidance in completing this study and for providing the base template.

1. Silvia Bucci, Mohammad Reza Loghmani, & Tatiana Tommasi. (2020). On the Effectiveness of Image Rotation for Open Set Domain Adaptation.

