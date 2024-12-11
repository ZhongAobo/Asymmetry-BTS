# Asymmetry-BTS
The original code used in this project comes from [RFNet](https://github.com/dyh127/RFNet), [RobustSeg](https://github.com/cchen-cc/Robust-Mseg), and [U-HVED](https://github.com/ReubenDo/U-HVED). The original projects for [RobustSeg](https://github.com/cchen-cc/Robust-Mseg) and [U-HVED](https://github.com/ReubenDo/U-HVED) are based on the TensorFlow framework. For ease of processing, we have rewritten these two projects using the PyTorch framework.
## Dataset
The dataset is sourced from [RFNet](https://github.com/dyh127/RFNet).

## Usage

1.Modify the parameters in ```train_ours.sh``` for each project, then run ```bash train_ours.sh``` to perform pre-training, fine-tuning, and post-training sequentially.

2.To run the baseline for each project, run ```bash train_baseline.sh```.

3.To evaluate a specific model weight, run ```bash eval.sh```.

4.If you want to generate a difference image separately, modify the datapath in line 9 of ```create_diff.py``` after obtaining the dataset, then run ```python create_diff.py```. The difference image will be available in the ```diff``` folder within the dataset directory.
