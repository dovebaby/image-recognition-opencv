SIFT+BoF+SVMによる一般物体認識
====

## Description
コンピュータビジョン・機械学習のライブラリであるOpenCVを使用した
SIFT・BoF・SVMによる一般物体認識プログラムです。

## Demo
元画像
![Original Image](/examples/frame_0.png)
二値化
![Original Image](/examples/frame_0_binary_c6.png)
輪郭検出
![Original Image](/examples/frame_0_contours.png)
凸包近似
![Original Image](/examples/frame_0_contours_convex.png)
切り抜き
![Original Image](/examples/frame_0_0_cup.png)



![Original Image](/examples/frame_0_0_SIFT_B.png)
![Original Image](/examples/frame_0_0_SIFT_G.png)
![Original Image](/examples/frame_0_0_SIFT_R.png)

## Requirement

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
