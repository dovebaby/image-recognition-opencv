SIFT+BoF+SVMによる一般物体認識
====

## Description
コンピュータビジョン・機械学習のライブラリであるOpenCVを使用した
SIFT・BoF・SVMによる一般物体認識プログラムです。

## Demo
二値化+輪郭検出による領域分割

|元画像|二値化+輪郭検出|切り抜き|  
|----|:----:|----|  
|![Original](/examples/frame_0.png)  |![Binary](/examples/frame_0_contours.png) |![Clipped](/examples/frame_0_0_cup.png)| 

SIFTをRGB毎に抽出  

|B|G|R|  
|----|:----:|----|  
|![B](/examples/frame_0_0_SIFT_B.png) |![G](/examples/frame_0_0_SIFT_G.png) |![R](/examples/frame_0_0_SIFT_R.png)|  




## Requirement
OpenCV 2.4.9  
Visual Studio 2013 Community  

## Usage
main.cpp
```cpp
# 一般物体認識（画像分類）
  createBOWCodebook(codebookFilename, "./Train", 50);
  convertImageToBOW(codebookFilename, "./Train", 100, trainDataFilename);
  convertImageToBOW(codebookFilename, "./Test", 100, testDataFilename);
  trainClassifier(trainDataFilename, classifierFilename, "train_results.txt");
  testClassifier(testDataFilename, classifierFilename, "test_results.txt");

# 二値化＋輪郭検出で領域分割を行った後各領域を認識
  recognizeImage(codebookFilename, trainDataFilename, classifierFilename, image, false);
```
