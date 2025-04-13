# FloodMask
Using before and after images from Copernicus Sentinel-1, a program has been developed to generate flood masks for affected regions.

Usage

1) Inference
	a) Folder Structure should be like: 
		project/
			│
			├── inference_flood.ipynb
			├── best_unet_model.keras
			├── images/
			│   ├── test_before/
			│   │   ├── img1.png
			│   │   ├── img2.png
			│   └── test_after/
			│       ├── img1.png
			│       ├── img2.png
			└── predicted_masks/

	b) Install packages in requirement.txt
	c) Run inference_flood.ipynb		
	
2) Training
	a)In Kaggle environment
		1) If you want to use Combined Drive dataset Run # flooddatsetmaker.ipynb with datasets: https://kaggle.com/datasets/1aab6e98f15ced12f05fd89457ef02f0c8f8235875210d821ca9fe918e1d8694 and https://kaggle.com/datasets/30c45215e79c2d5e216b8bc3e7b8f15558ffd6743380d5a02a435513a0611d6b, get zip file save as floodmapping2 dataset in kaggle,
		 then run # floodunetv1.ipynb
		2) Else, use https://kaggle.com/datasets/b245db8c6f4b36c861d85888c9a14153aeca50dbac50354cede7d27724a14fa8, then run # floodunetv1.ipynb
		
		
