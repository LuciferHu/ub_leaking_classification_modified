# ub_leaking_classification_modified
This project based on the repository [environmental-sound-classification](https://github.com/mariostrbac/environmental-sound-classification) and [UrbanSound8K_OfflineDataAugmentation](https://github.com/LuciferHu/UrbanSound8K_OfflineDataAugmentation).

The main processing pipeline of this project basically followed the method used in [environmental-sound-classification]. What I have done is to change the project to python scripts and add other CNN models to the project. The project has been executed in Pycharm.

Note that raw data is from [UrbanSound8K_OfflineDataAugmentation] and they have been converted to melspectrogram, stacked in a pickle file in advance, which is confidential.

# dependecies
torch >=1.8.0
torchvision >= 0.9.0

# Usage
There are 5 python scripts and 1 python package in this project.
- `data_aug`:
     - on-line data augmentation, shift the melspectrogram to the right with a certain probability and unsqueeze the channel dimension.

- `data_set`:
     - inherited by data.Dataset, the main dataset class.
      
- `model_select`:
     - every epoch of train and evaluation is performed in here.

- `train`:
     - the main train file. You need to specify the model name, epoch num, the repeat num for every whole train process and of course, the fold name for valuation.
     - history will be stacked in trend.csv saved in diractory "log", in name order.
      
- `utils`:
     - normalize data and plot results.
