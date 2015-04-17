#include "image_recog.h"

int recognizeImage(string codebookFilename, string dataFilename, string classifierFilename, Mat &inputImage, bool useCaptureDevice) {

  // キャプチャーデバイスの初期化
  Ptr<VideoCapture> capture;
  if (useCaptureDevice) {
    cout << "Initializing the capture device..." << endl;
    int device = 0;
    capture = Ptr<VideoCapture>(new VideoCapture(device));
    if (!capture->isOpened()) {
      cerr << "Error: Could not open the capture device." << endl;
      return -1;
    }
  }

  // カテゴリー名の読み込み
  vector<string> classNames;
  FileStorage cvFileStorageRead2(dataFilename, FileStorage::READ);
  if (!cvFileStorageRead2.isOpened()) {
    cerr << "Error: Could not open the data." << endl;
    return -1;
  }
  cvFileStorageRead2["ClassName"] >> classNames;
  classNames.insert(classNames.begin(), "");  // ラベル対応付け調整

  // SIFTFeatureDetector, Extractorの設定
  class Feature feature;

  // BOW特徴抽出器パラメータ設定
  class BOW bow;
  if (!USE_COLOR_FEATURE) {
    bow.setDExtractor(codebookFilename);
  }
  else {
    bow.setDExtractorBGR(codebookFilename);
  }

  // 分類器の設定
  class ClassifierSVM classifier;
  classifier.load(classifierFilename);

  // フレーム処理
  cout << "Press e: execute, q: quit" << endl;
  Mat frame;
  vector<Mat> resultFrames;
  vector<Mat> resultFramesSIFT;
  vector<vector<Mat>> resultFramesSIFTBGR;
  vector<int> resultClassnames;
  int saveImageCount = 0;

  // トラックバー
  int barBlockSize = 11;
  int barC = 5;
  int binarizeMethod = 1;
  int invertImage = 1;
  namedWindow("Config", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
  createTrackbar("BlockSize", "Config", &barBlockSize, 101);
  createTrackbar("C", "Config", &barC, 100);
  createTrackbar("Method", "Config", &binarizeMethod, 2);
  createTrackbar("Invert", "Config", &invertImage, 1);

  for (;;) {

    if (useCaptureDevice) {
      *capture >> frame;  // キャプチャーデバイスから読み込み
    }
    else {
      frame = inputImage; // ファイルから読み込み
    }

    if (frame.empty()) {
      cerr << "Error: Could not open the image." << endl;
      return -1;
    }
    imshow("Capture", frame);

    // キー入力
    char c = waitKey(100);
    if (c == 'e') {

      // グレースケール画像の2値化
      Mat grayImage, binaryImage;
      cvtColor(frame, grayImage, CV_BGR2GRAY);

      // 大津の手法
      double thresh = 0.;
      Mat histogram;
      if (binarizeMethod == 0) {
        thresh = threshold(grayImage, binaryImage, .0, 255.0, THRESH_BINARY | THRESH_OTSU);
        showHistogram(grayImage, histogram, thresh); // ヒストグラム
      }

      // Adaptive Threshold
      int blockSize = barBlockSize; // blockSize x blockSize近傍（必ず奇数）
      if (blockSize % 2 == 0) {
        blockSize += 1;
        cout << "Warning: BlockSize must be an odd number." << endl;
      }
      double cAdaptive = barC; // 減算定数
      if (binarizeMethod == 1) {
        const int adaptiveMethod = ADAPTIVE_THRESH_MEAN_C; // MEAN（平均） / GAUSSIAN（重み付け）
        adaptiveThreshold(grayImage, binaryImage, 255.0, adaptiveMethod, CV_THRESH_BINARY, blockSize, cAdaptive);
      }

      // 2値化なし
      if (binarizeMethod == 2) {
        binaryImage = Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar::all(0));
      }

      imshow("Binary", binaryImage);

      // 輪郭を抽出
      vector<vector<Point>> contours;
      vector<Vec4i> hierarchy;
      Mat binaryImageSave;
      binaryImage.copyTo(binaryImageSave);
      if (invertImage == 1) {
        binaryImage = ~binaryImage; // 輪郭検出のため反転
      }
      findContours(binaryImage, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE); // !入力画像書き換え注意
      Mat frameContourSave(Size(frame.cols, frame.rows), CV_8UC3, Scalar::all(255));

      // 輪郭描画
      for (unsigned i = 0; i < contours.size(); i++) {
        Scalar color(rand() & 255, rand() & 255, rand() & 255);
        drawContours(frameContourSave, contours, i, color, 1);
      }
      imshow("Contour", frameContourSave);
      Mat frameContourConvexSave;
      frameContourSave.copyTo(frameContourConvexSave);

      // 各領域の処理
      Mat frameResultsSave;
      frame.copyTo(frameResultsSave);
      for (unsigned i = 0; i < contours.size(); i++) {

        // 過剰に小さい領域は処理しない
        if (contours.at(i).size() < 50) {
          continue;
        }
        if (contourArea(contours.at(i), false) < 1000) {
          continue;
        }

        // マスク画像作成
        vector<Point> convex;
        convexHull(contours.at(i), convex); // 輪郭を囲う凸包を計算
        Mat frameMask(Size(frame.cols, frame.rows), CV_8UC3, Scalar::all(255));
        fillConvexPoly(frameMask, convex, Scalar::all(0)); // 凸包のみ残すマスク画像

        // マスク処理
        Mat frameMasked;
        frame.copyTo(frameMasked);
        frameMasked += frameMask; // マスク処理
        //imshow("frameMasked", frameMasked); // バグ確認用

        // SIFT検出用マスク画像（非ゼロ要素が検出領域）
        vector<Mat> siftMask;
        split(frameMask, siftMask);
        //imshow("siftMask", ~siftMask.at(0));
        //waitKey();

        // 領域から特徴量を抽出
        Mat bowDescriptors;
        vector<KeyPoint> keypoints;
        vector<vector<KeyPoint>> keypointsBGR(3);
        if (!USE_COLOR_FEATURE) {
          feature.detectKeypointsWithMask(frame, ~siftMask.at(0), keypoints);
          bow.descript(frame, keypoints, bowDescriptors);
        } // 1チャンネル使用時
        else{
          vector<Mat> bowDescriptorsBGR(3);
          feature.detectKeypointsBGRWithMask(frame, ~siftMask.at(0), keypointsBGR);
          bow.descriptBGR(frame, keypointsBGR, bowDescriptorsBGR, bowDescriptors);
        } // 3チャンネル使用時

        // キーポイント検出無し・BOWの次元不足時は次輪郭の処理へ
        if (!USE_COLOR_FEATURE && (bowDescriptors.cols != VISUAL_WORDS_COUNT)) {
          continue;
        }
        if (USE_COLOR_FEATURE && (bowDescriptors.cols != VISUAL_WORDS_COUNT * 3)) {
          continue;
        }

        // 領域を識別
        Mat results;
        classifier.predict(bowDescriptors, results);

        // 結果表示用
        if (!USE_COLOR_FEATURE) {
          Mat frameSIFT;
          drawKeypoints(frame, keypoints, frameSIFT, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
          resultFramesSIFT.push_back(frameSIFT);
        }
        else{
          vector<Mat> frameBGR;
          vector<Mat> frameSIFTBGR(3);
          split(frame, frameBGR);
          drawKeypoints(frameBGR.at(0), keypointsBGR.at(0), frameSIFTBGR.at(0), Scalar(255, 0, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
          drawKeypoints(frameBGR.at(1), keypointsBGR.at(1), frameSIFTBGR.at(1), Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
          drawKeypoints(frameBGR.at(2), keypointsBGR.at(2), frameSIFTBGR.at(2), Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
          resultFramesSIFTBGR.push_back(frameSIFTBGR);
        }
        polylines(frameContourConvexSave, convex, true, Scalar(0, 0, 0), 2);   // 保存用

        // キーポイント数の少ない領域を除外
        if (!USE_COLOR_FEATURE && keypoints.size() < 10) {
          polylines(frameResultsSave, convex, true, Scalar(255, 0, 0), 1);
        }
        else if (USE_COLOR_FEATURE && (keypointsBGR.at(0).size() < 10 || keypointsBGR.at(1).size() < 10 || keypointsBGR.at(2).size() < 10)) {
          polylines(frameResultsSave, convex, true, Scalar(255, 0, 0), 1);
        }
        // キーポイントが一定数あれば領域を識別
        else {
          polylines(frameResultsSave, convex, true, Scalar(0, 0, 255), 2);
          Point convexPtsTotal;
          for (unsigned j = 0; j < convex.size(); j++) {
            convexPtsTotal += convex.at(j);
          }
          Point2f textPoint = Point2f(convexPtsTotal.x / static_cast<float> (convex.size()), convexPtsTotal.y / static_cast<float> (convex.size()));
          putText(
            frameResultsSave,
            classNames.at(static_cast<int> (results.at<float>(0, 0))),
            textPoint,
            FONT_HERSHEY_SIMPLEX,
            1,
            Scalar(0, 0, 255),
            2
            );
        }

        resultClassnames.push_back(static_cast<int> (results.at<float>(0, 0)));
        resultFrames.push_back(frameMasked); // 結果を保持

      } // 各領域の処理ここまで
      imshow("Results", frameResultsSave);

      // 結果を保存
      cout << "Save the results? (y/n)" << endl;
      c = waitKey();

      if (c == 'y') {

        imwrite("frame_" + to_string(saveImageCount) + ".png", frame); // 元画像

        // 2値化画像
        if (binarizeMethod == 0) {
          imwrite("frame_" + to_string(saveImageCount) + "_binary_t" + to_string(thresh) + ".png", binaryImageSave);
        }
        else if (binarizeMethod == 1) {
          imwrite("frame_" + to_string(saveImageCount) + "_binary_c" + to_string(cAdaptive) + ".png", binaryImageSave);
        }
        else {
          imwrite("frame_" + to_string(saveImageCount) + "_binary.png", binaryImageSave);
        }

        imwrite("frame_" + to_string(saveImageCount) + "_contours.png", frameContourSave); // 輪郭
        imwrite("frame_" + to_string(saveImageCount) + "_contours_convex.png", frameContourConvexSave); // 輪郭+凸包
        imwrite("frame_" + to_string(saveImageCount) + "_result.png", frameResultsSave); // 識別結果

        // 各領域の画像
        for (unsigned i = 0; i < resultFrames.size(); i++) {
          imwrite("frame_" + to_string(saveImageCount) + "_" + to_string(i) + "_" + classNames.at(resultClassnames.at(i)) + ".png", resultFrames.at(i));
        }
        if (!USE_COLOR_FEATURE) {
          for (unsigned i = 0; i < resultFramesSIFT.size(); i++) {
            imwrite("frame_" + to_string(saveImageCount) + "_" + to_string(i) + "_SIFT.png", resultFramesSIFT.at(i));
          }
        }
        else {
          for (unsigned i = 0; i < resultFramesSIFTBGR.size(); i++) {
            imwrite("frame_" + to_string(saveImageCount) + "_" + to_string(i) + "_SIFT_B.png", resultFramesSIFTBGR.at(i).at(0));
            imwrite("frame_" + to_string(saveImageCount) + "_" + to_string(i) + "_SIFT_G.png", resultFramesSIFTBGR.at(i).at(1));
            imwrite("frame_" + to_string(saveImageCount) + "_" + to_string(i) + "_SIFT_R.png", resultFramesSIFTBGR.at(i).at(2));
          }
        }

        // 輝度値ヒストグラム（大津の方法）
        if (binarizeMethod == 0) {
          ofstream fout("frame_" + to_string(saveImageCount) + "_hist.txt", ios::out);
          fout << histogram << endl;
        }

        cout << "Saved the results." << endl;
        saveImageCount++;

      }

      else {
        cout << "Cleared the results." << endl;
      }

      resultFrames.clear(); // 保持した結果を削除
      resultFramesSIFT.clear();
      resultFramesSIFTBGR.clear();
      resultClassnames.clear();

    }

    else if (c == 'q') {
      break;
    }

  }

  return 0;
}

int createBOWCodebook(string codebookFilename, string dataDirectory, int imageCount) {

  cout << "Creating a bow codebook..." << endl;
  class Feature feature;

  // BOWTrainer設定
  class BOW bow;
  if (!USE_COLOR_FEATURE) {
    bow.setBOWTrainer();
  }
  else {
    bow.setBOWTrainerBGR();
  }

  // データセットの読み込み
  int sampleCount = 0;
  vector<int> keypointTotal(3, 0);
  fsys::path p(dataDirectory);
  for_each(
    fsys::recursive_directory_iterator(p),
    fsys::recursive_directory_iterator(),
    // 参照によりキャプチャ
    [&](fsys::path p) {

    // ディレクトリの場合
    if (fsys::is_directory(p)) {
      cout << "directory: " << p.leaf() << endl;
      sampleCount = 0;
    }

    // 1カテゴリーBOW_TRAIN_COUNT枚まで読み込み
    else if (sampleCount < imageCount) {

      // 画像の読み込み
      Mat image = imread(p.string(), 1);
      if (image.empty()) {
        cerr << "Error: Could not open one of the images." << endl;
      }

      // 成功の場合
      else {

        if (!USE_COLOR_FEATURE) {

          vector<KeyPoint> keypoints;
          Mat descriptors;
          feature.detectAndDescript(image, keypoints, descriptors); // SIFT特徴の計算
          cout << p.leaf() << ": " << keypoints.size() << endl; // キーポイントの数を表示
          keypointTotal.at(0) += keypoints.size(); // 総キーポイント数カウント
          bow.addToBOWTrainer(descriptors); // BOWTrainerに特徴ベクトルを追加

        }
        // 1チャンネル使用時↑

        // 3チャンネル使用時↓
        else {

          vector<vector<KeyPoint>> keypointsBGR(3);
          vector<Mat> descriptorsBGR(3);
          feature.detectAndDescriptBGR(image, keypointsBGR, descriptorsBGR);
          cout << p.leaf() << ":  " << keypointsBGR.at(0).size() << "  "
            << keypointsBGR.at(1).size() << "  " << keypointsBGR.at(2).size() << endl;
          for (int i = 0; i < 3; i++) {
            keypointTotal.at(i) += keypointsBGR.at(i).size();
            bow.addToBOWTrainerBGR(descriptorsBGR);
          }

        }

        sampleCount++;

      }
    }
  }
  );


  // コードブックの作成
  if (!USE_COLOR_FEATURE) {
    bow.createCodebook(codebookFilename, keypointTotal.at(0));
  }
  else {
    bow.createCodebookBGR(codebookFilename, keypointTotal);
  }


  return 0;
}

int convertImageToBOW(string codebookFilename, string dataDirectory, int conutCreateBOW, string dataFilename) {

  // BOW特徴抽出器パラメータ設定
  class BOW bow;
  if (!USE_COLOR_FEATURE) {
    bow.setDExtractor(codebookFilename);
  }
  else {
    bow.setDExtractorBGR(codebookFilename);
  }

  // 特徴抽出器パラメータ設定
  class Feature feature;

  // データセットの読み込み
  int classLabel = 0;
  int sampleCount = 0;
  Mat keypointCount;
  vector<string> classNames;
  class Dataset data;

  fsys::path p(dataDirectory);
  for_each(
    fsys::recursive_directory_iterator(p),
    fsys::recursive_directory_iterator(),
    [&](fsys::path p) {

    // ディレクトリの場合
    if (fsys::is_directory(p)) {
      classNames.push_back(p.leaf());
      classLabel++;
      cout << "directory: " << p.leaf() << " (label: " << (classLabel) << ")" << endl;
      sampleCount = 0;
    }

    // 1カテゴリーCREATE_BOW_COUNT枚まで読み込み
    else if (sampleCount < conutCreateBOW) {

      // 画像の読み込み
      Mat image = imread(p.string(), 1);
      if (image.empty()) {
        cerr << "Error: Could not open one of the images." << endl;
      }

      // 成功の場合
      else {

        Mat bowDescriptors;

        if (!USE_COLOR_FEATURE) {

          // キーポイントの計算
          vector<KeyPoint> keypoints;
          feature.detectKeypoints(image, keypoints);
          cout << p.leaf() << ":  " << keypoints.size() << endl;

          // BOWの計算
          bow.descript(image, keypoints, bowDescriptors);

        }
        // 1チャンネル使用時↑


        // 3チャンネル使用時↓
        else {

          vector<vector<KeyPoint>> keypointsBGR(3);
          vector<Mat> bowDescriptorsBGR(3);
          feature.detectKeypointsBGR(image, keypointsBGR);
          cout << p.leaf() << ":  " << keypointsBGR.at(0).size() << "  "
            << keypointsBGR.at(1).size() << "  " << keypointsBGR.at(2).size() << endl;
          bow.descriptBGR(image, keypointsBGR, bowDescriptorsBGR, bowDescriptors);

        }

        // Features, Responses, Filenameにデータを追加
        data.features.push_back(bowDescriptors);
        data.responses.push_back(classLabel);
        data.filenames.push_back(p.leaf());

        sampleCount++;
      }
    }
  });

  // 行列featuresのサイズを確認
  cout << "Mat features: " << data.features.rows << "x" << data.features.cols << endl;
  cout << "Mat responses: " << data.responses.rows << "x" << data.responses.cols << endl;

  // データの保存
  FileStorage cvFileStorageWrite(dataFilename, FileStorage::WRITE);
  write(cvFileStorageWrite, "ClassName", classNames);
  write(cvFileStorageWrite, "Filename", data.filenames);
  write(cvFileStorageWrite, "BOW", data.features);
  write(cvFileStorageWrite, "Responses", data.responses);
  cout << "Saved the bow and responses." << endl;

  return 0;
}

int trainClassifier(string dataFilename, string classifierFilename, string resultFilename) {

  // データの読み込み
  cout << "Loading the bow and responses..." << endl;
  vector<string> classNames;
  class Dataset train;
  FileStorage cvFileStorageRead(dataFilename, FileStorage::READ);
  if (!cvFileStorageRead.isOpened()) {
    cerr << "Error: Could not open the data." << endl;
    return -1;
  }
  cvFileStorageRead["ClassName"] >> classNames;
  classNames.insert(classNames.begin(), "");  // ラベル対応付け調整
  cvFileStorageRead["Filename"] >> train.filenames;
  cvFileStorageRead["BOW"] >> train.features;
  cvFileStorageRead["Responses"] >> train.responses;

  // データサイズのチェック
  if (train.features.empty()) {
    cerr << "Error: Could not load the data." << endl;
    return -1;
  }
  cout << "Mat train.features: " << train.features.rows << "x" << train.features.cols << endl;
  cout << "Mat train.responses: " << train.responses.rows << "x" << train.responses.cols << endl;

  // 分類器で学習
  cout << "Training the classifier..." << endl;
  class ClassifierSVM classifier;
  classifier.build(train.features, train.responses, classifierFilename);

  // 結果の保存先を確保
  ofstream fout(resultFilename, ios::app);
  fout << "BOW特徴ベクトル: " << train.features.cols << "次元\n" <<
    "データ数: " << train.features.rows << "\n" <<
    "クラス数: " << classNames.size() - 1 << endl;

  // 訓練誤差を計算
  Mat results;
  classifier.predict(train.features, results);
  int trainError = 0;
  for (int i = 0; i < results.rows; i++) {
    //results.depth()=CV_32F, trainResponses.depth()=CV_32S
    if (train.responses.at<signed int>(i, 0) != results.at<float>(i, 0)) {
      trainError++;
    }
  }
  fout << "Error(resub): " << trainError * 100 / static_cast<double> (train.features.rows) << "%" << endl;
  cout << "Error(resub): " << trainError * 100 / static_cast<double> (train.features.rows) << "%" << endl;

  return 0;
}

int testClassifier(string dataFilename, string classifierFilename, string resultFilename) {

  // データの読み込み
  cout << "Loading the bow and responses..." << endl;
  vector<string> classNames;
  class Dataset test;
  Mat keypointCount;
  FileStorage cvFileStorageRead(dataFilename, FileStorage::READ);
  if (!cvFileStorageRead.isOpened()) {
    cerr << "Error: Could not open the data." << endl;
    return -1;
  }
  cvFileStorageRead["ClassName"] >> classNames;
  cvFileStorageRead["Filename"] >> test.filenames;
  cvFileStorageRead["BOW"] >> test.features;
  cvFileStorageRead["Responses"] >> test.responses;
  if (test.features.empty()) {
    cerr << "Error: Could not load the data." << endl;
    return -1;
  }

  // ホールドアウト法で誤識別率を推定
  classNames.insert(classNames.begin(), "");  // ラベル対応付け調整
  vector<int> testErrors(CLASS_COUNT + 1, 0); // カテゴリー毎に誤識別をカウント
  vector<int> keypoints(CLASS_COUNT + 1, 0);
  int sampleCount = 0;
  cout << "Mat test.features: " << test.features.rows << "x" << test.features.cols << endl;
  cout << "Mat test.responses: " << test.responses.rows << "x" << test.responses.cols << endl;

  // 分類器の設定
  class ClassifierSVM classifier;
  classifier.load(classifierFilename);

  // 結果の保存先を確保
  ofstream fout(resultFilename, ios::app);
  fout << "BOW特徴ベクトル: " << test.features.cols << "次元\n" <<
    "データ数: " << test.features.rows << "\n" <<
    "クラス数: " << classNames.size() - 1 << endl;

  // 識別
  Mat results;
  classifier.predict(test.features, results);
  for (int i = 0; i < test.features.rows; i++) {
    if (test.responses.at<signed int>(i, 0) != results.at<float>(i, 0)) {
      fout << classNames.at(test.responses.at<signed int>(i, 0)) << "\\" << test.filenames.at(i) <<
        " is misclassified as " << classNames.at(static_cast<int> (results.at<float>(i, 0))) << "." << endl;
      //cout << classNames.at(test.responses.at<signed int>(i,0)) << "\\" << test.filenames.at(i) <<
      //  " is misclassified as " << classNames.at(static_cast<int> (results.at<float>(i,0))) << "." << endl;
      testErrors.at(test.responses.at<signed int>(i, 0))++;
    }
    else {
      fout << classNames.at(test.responses.at<signed int>(i, 0)) << "\\" << test.filenames.at(i) <<
        " is correctly classified." << endl;
      //cout << classNames.at(test.responses.at<signed int>(i,0)) << "\\" << test.filenames.at(i) <<
      //  " is correctly classified." << endl;
    }
  }

  // 汎化誤差を計算
  int errorTotal = 0;
  for (int i = 0; i < CLASS_COUNT; i++) {
    fout << "Error(" << classNames.at(i + 1) << "): " <<
      testErrors.at(i + 1) * 100 * CLASS_COUNT / static_cast<double> (test.features.rows) << "%" << endl;
    cout << "Error(" << classNames.at(i + 1) << "): " <<
      testErrors.at(i + 1) * 100 * CLASS_COUNT / static_cast<double> (test.features.rows) << "%" << endl;
    errorTotal += testErrors.at(i + 1);
  }
  fout << "Error(total): " << errorTotal * 100 / static_cast<double> (test.features.rows) << "%" << endl;
  cout << "Error(total): " << errorTotal * 100 / static_cast<double> (test.features.rows) << "%" << endl;

  return 0;
}

//http://docs.opencv.org/doc/tutorials/imgproc/histograms/histogram_calculation/histogram_calculation.html
void showHistogram(Mat &grayImage, Mat &histogram, double &thresh) {

  /// Separate the image in 3 places ( B, G and R )
  //vector<Mat> bgr_planes;
  //split(src, bgr_planes);

  /// Establish the number of bins
  int histSize = 256;

  /// Set the ranges ( for B,G,R) )
  float range[] = { 0, 256 };
  const float* histRange = { range };

  bool uniform = true; bool accumulate = false;

  //Mat b_hist, g_hist, r_hist;

  /// Compute the histograms:
  calcHist(&grayImage, 1, 0, Mat(), histogram, 1, &histSize, &histRange, uniform, accumulate);
  //calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
  //calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

  // Draw the histograms for B, G and R
  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound((double)hist_w / histSize);

  Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

  /// Normalize the result to [ 0, histImage.rows ]
  normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat());
  //normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
  //normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

  /// Draw for each channel
  for (int i = 1; i < histSize; i++)
  {
    line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(histogram.at<float>(i - 1))),
      Point(bin_w*(i), hist_h - cvRound(histogram.at<float>(i))),
      Scalar(255, 255, 255), 2, 8, 0);
    /*line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
      Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
      Scalar(0, 255, 0), 2, 8, 0);*/
    /*line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
      Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
      Scalar(0, 0, 255), 2, 8, 0);*/
  }

  /// Draw threshold
  int threshPoint = saturate_cast<int>(bin_w * thresh);
  line(histImage, Point(threshPoint, 0), Point(threshPoint, histImage.rows), Scalar(0, 0, 255), 2);

  /// Display
  namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
  imshow("calcHist Demo", histImage);

}
