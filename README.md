# ROS_AMLProject-5


## About

The project is a simplified version Open-Set Domain Adaptation through Self-Supervision.

## Getting Started

### Prerequisites

All prerequisites can be found in [requirements.txt](requirements.txt).

## Usage

Simply run `main.py` with the desired arguments:

- --load_weights: used to load stored weights of the model
- --source: source domain
- --target: target domain
- --n_classes_known: number of known classes
- --n_classes_tot: number of classes included unknown
- --path_dataset: path where the dataset is located
- --min_scale: minimum scale percent
- --max_scale: maximum scale percent
- --jitter: jitter to apply
- --random_grayscale: probability of greyscaling the image
- --image_size: input size of the model (used to resize images)
- --batch_size: batch size of the model
- --learning_rate: learning rate of the model
- --epochs_step1: number of epochs of step1 for known/unknown separation
- --epochs_step2: number of epochs of step2 for source-target adaptation
- --train_classifiers: used to freeze all except the classifiers
- --weight_RotTask_step1: weight for the rotation loss in step1
- --weight_RotTask_step2: weight for the rotation loss in step2
- --threshold: threshold for the known/unkown separation
- --steps: choose which steps to run

## Authors & contributors

For a full list of all authors and contributors, see [the contributors page](https://github.com/DarthReca/AML-Project/contributors).

## License

This project is licensed under the **MIT license**.

See [LICENSE](LICENSE) for more information.

## Acknowledgements

Thanks to Silvia Bucci for the support in realizing the project and for providing the base template.

1. Silvia Bucci, Mohammad Reza Loghmani, & Tatiana Tommasi. (2020). On the Effectiveness of Image Rotation for Open Set Domain Adaptation.

