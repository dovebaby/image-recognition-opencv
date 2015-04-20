SIFT+BoF+SVMによる一般物体認識
====

## Description
コンピュータビジョン・機械学習のライブラリであるOpenCVを使用した
SIFT・BoF・SVMによる一般物体認識プログラムです。

## Demo
* 二値化+輪郭検出による領域分割

|元画像|二値化+輪郭検出|切り抜き|  
|----|:----:|----|  
|![Original](/examples/frame_0.png)  |![Binary](/examples/frame_0_contours.png) |![Clipped](/examples/frame_0_0_cup.png)|  
  
* SIFTをRGB毎に抽出  

|B|G|R|  
|----|:----:|----|  
|![B](/examples/frame_0_0_SIFT_B.png) |![G](/examples/frame_0_0_SIFT_G.png) |![R](/examples/frame_0_0_SIFT_R.png)|  
  
* 認識例  
![EX](/examples/frame_1_result.png)  

## Requirement
1. Visual Studio 2013 Community  
　学習用画像の読み込み（`#include <filesystem>`を使用）のためVS2012以上が必要  
  
2. OpenCV 2.4.9  
　以下のサイトを参考にOpenCVをインストール・VSにパスを通す  
　[OpenCV を Visual Studio で使う方法](http://physics-station.blogspot.jp/2013/03/opencv-visual-studio.html)  
  
## Usage
* main.cpp  
学習・テスト画像のフォルダのパスを引数で指定する。  
指定した場所にクラス毎にフォルダを分けて画像を置く（フォルダ名=クラス名）  
```cpp
// 一般物体認識（画像分類）
  createBOWCodebook(codebookFilename, "./Train", 50);
  convertImageToBOW(codebookFilename, "./Train", 100, trainDataFilename);
  convertImageToBOW(codebookFilename, "./Test", 100, testDataFilename);
  trainClassifier(trainDataFilename, classifierFilename, "train_results.txt");
  testClassifier(testDataFilename, classifierFilename, "test_results.txt");
  
// 二値化＋輪郭検出で領域分割を行った後各領域を認識
  recognizeImage(codebookFilename, trainDataFilename, classifierFilename, image, false);
```
  
* image_recog.h
使用する特徴量、認識対象とするクラス数、BoFの次元を設定
```cpp
  const string FEATURE_DETECTOR_TYPE = "SIFT";      // SIFT, Dense, GridSIFT, SURF, DynamicSURF
  const string DESCRIPTOR_EXTRACTOR_TYPE = "SIFT";  // SIFT, SURF
  const bool USE_COLOR_FEATURE = true;              // BGR各成分から特徴抽出
  const int CLASS_COUNT = 10;                       // クラス数
  const int VISUAL_WORDS_COUNT = 2000;              // BOW特徴ベクトルの次元 (RGB:1成分あたり)
```

