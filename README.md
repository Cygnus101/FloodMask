# FloodMask

Using before and after images from Copernicus Sentinel-1, a program has been developed to generate flood masks for affected regions.

## Usage

### 1) Inference

#### a) Folder Structure

project/ <br>
├── inference_flood.ipynb <br>
├── best_unet_model.keras <br>
├── images/ <br>
│   ├── test_before/ <br>
│   │   ├── img1.png <br>
│   │   └── img2.png <br>
│   └── test_after/ <br>
│       ├── img1.png <br>
│       └── img2.png <br>
└── predicted_masks/ <br>


#### b) Install packages

Install dependencies from `requirements.txt`:


#### c) Run inference

Open and run `inference_flood.ipynb`.

---

### 2) Training

#### a) In Kaggle Environment

1. **Using Combined Drive Dataset**  
   - Use the following datasets:  
     - https://kaggle.com/datasets/1aab6e98f15ced12f05fd89457ef02f0c8f8235875210d821ca9fe918e1d8694  
     - https://kaggle.com/datasets/30c45215e79c2d5e216b8bc3e7b8f15558ffd6743380d5a02a435513a0611d6b  
   - Save them together as a dataset named `floodmapping2` in your Kaggle account.  
   - Run `flooddatasetmaker.ipynb`, then `floodunetv1.ipynb`.

2. **Using Single Dataset**  
   - Use this dataset:  
     - https://kaggle.com/datasets/b245db8c6f4b36c861d85888c9a14153aeca50dbac50354cede7d27724a14fa8  
   - Run `floodunetv1.ipynb` directly.

		
		
