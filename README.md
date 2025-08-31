# Graph-Based Uncertainty Modeling and Multimodal Fusion for Salient Object Detection
We propose **DUP-MCRNet**, a novel salient object detection framework based on dynamic uncertainty propagation and multimodal collaborative reasoning.

## Paper
🎉 **Congratulations!** Our paper has been accepted at the <span style="color: #0066cc;">**[32nd International Conference on Neural Information Processing (ICONIP 2025)](https://iconip2025.apnns.org/)**, Okinawa, Japan </span> ✨
You can check the manuscript on [Arxiv](http://arxiv.org/abs/2508.20415).

<img width="3906" height="2227" alt="ICONIP 2025 figure" src="https://github.com/user-attachments/assets/03abd6b3-65ca-4696-9022-137ab9d5f7e9" />

## Environment

Our project supports **Python 3.9.13** and **PyTorch 1.11.0**. Other required packages can be installed via:

```bash
pip install -r requirements.txt
```

## Data Preparation

All datasets used can be downloaded from [here](https://pan.baidu.com/s/1fw4uB6W8psX7roBOgbbXyA) (password: `arrr`).
*Note: The datasets are provided by the other author.*

After downloading, put them into `datasets/` folder. Your `datasets/` folder should be looked like this:

````
-- datasets
   |-- DUT-O
   |   |--imgs
   |   |--gt
   |-- DUTS-TR
   |   |--imgs
   |   |--gt
   |-- ECSSD
   |   |--imgs
   |   |--gt
   ...
````

## Pretrained Weights Preparation
Before training, please download the pretrained backbone weights and place them into the `pretrained_model/` folder.  

Currently supported backbones:
- **ResNet**: [Download Link](https://pan.baidu.com/s/1JBEa06CT4hYh8hR7uuJ_3A) (password: `uxcz`) *Note: The pretrained weights are provided by the other author.*
- **SwinTransformer**: [GitHub Link](https://github.com/microsoft/Swin-Transformer) 

Make sure the weights are correctly placed before running the training script. Your `pretrained_model/` folder should look like this:

````
-- pretrained_model
   |-- resnet50.pth
   |
   |-- swin_base_patch4_window12_384_22k.pkl
````

## Training and Testing

After preparing the pretrained weights and datasets, you can train and test the model in one step:

```bash
python train_test.py --train=True --test=True --record='record.txt'
```

The predictions will be in `preds/` folder and the training records will be in `record.txt` file. 


## Evaluation

After training and testing, you can evaluate the model using the following tools:

- For **Precision-Recall (PR) curves**, we use the code provided by this repo: [Binary Segmentation Evaluation Tool](https://github.com/xuebinqin/Binary-Segmentation-Evaluation-Tool)  

- For **MAE, mSIOU, S-measure, and Weighted F-measure**, we use the code provided by this repo: [PySODMetrics](https://github.com/lartpang/PySODMetrics)  


## Evaluation Results
### Quantitative Evaluation
<img width="777" height="189" alt="table1" src="https://github.com/user-attachments/assets/6c7c938c-55a2-4839-a963-4584c4048f63" />

### Ablation Study
<img width="777" height="156" alt="table2" src="https://github.com/user-attachments/assets/94e2e080-a97a-4517-b414-adb8398b4ce4" />

### Precision-recall curve
<img width="3609" height="1799" alt="figure2" src="https://github.com/user-attachments/assets/ecf9195b-98c2-4e1b-888f-513506a007b6" />

### Visual Comparison
<img width="3506" height="2033" alt="figure3" src="https://github.com/user-attachments/assets/0d6df8b1-817d-4e7d-9eaf-1f4e031f8c8c" />

