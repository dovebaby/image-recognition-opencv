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

## Requirement
1. Visual Studio 2013 Community  
　学習用画像の読み込み（`#include <filesystem>`を使用）のためVS2012以上が必要  
  
2. OpenCV 2.4.9  
　以下のサイトを参考にOpenCVをインストール・VSにパスを通す  
　[OpenCV を Visual Studio で使う方法](http://physics-station.blogspot.jp/2013/03/opencv-visual-studio.html)  
  
