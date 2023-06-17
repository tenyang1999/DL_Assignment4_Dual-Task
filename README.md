# DL_Assignment4_Dual-Task

- 深度學習作業4：實作Dual-Task Deep Learning: Simultaneous Object Detection and Semantic Segmentation from Separate Datasets
- 使用VOC2007與ADE20K的資料集進行預測
- 透過mean average accuracy & mean aerage recall去評分，比較模型效能

## 安裝
- 本次執行上會需要用到的package

```python
pip install numpy 
pip install pandas
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

## 使用方法
- ADEdataset.py為ADE的dataset讀取型態
- VOC_detect.ipynb為利用VOC進行Object Detection
- ADE_seg.ipynb為利用ADE進行Semantic Segmentation
- dual_task.ipynb為將兩者進行整合同時訓練
- utils為放置訓練以及預測會用到的工具們
