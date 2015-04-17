#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <cstdio>
#include <filesystem> // VS2012以上
#include <opencv2/opencv.hpp>
#include "opencv2/opencv_lib.hpp"
#include "opencv2/nonfree/nonfree.hpp" // SIFT, SURFを使用する場合

using namespace cv;
using namespace std;
namespace fsys = std::tr2::sys;  // filesystem

const string FEATURE_DETECTOR_TYPE = "SIFT";      // SIFT, Dense, GridSIFT, SURF, DynamicSURF
const string DESCRIPTOR_EXTRACTOR_TYPE = "SIFT";  // SIFT, SURF
const bool USE_COLOR_FEATURE = true;              // BGR各成分から特徴抽出
const int CLASS_COUNT = 10;                       // クラス数
const int VISUAL_WORDS_COUNT = 2000;              // BOW特徴ベクトルの次元 (RGB:1成分あたり)

void showHistogram(Mat &grayImage, Mat &histogram, double &thresh);
int createBOWCodebook(string codebookFilename, string dataDirectory, int imageCount);
int convertImageToBOW(string codebookFilename, string dataDirectory, int conutCreateBOW, string dataFilename);
int trainClassifier(string dataFilename, string classifierFilename, string resultFilename);
int testClassifier(string dataFilename, string classifierFilename, string resultFilename);
int recognizeImage(string codebookFilename, string dataFilename, string classifierFilename, Mat &inputImage, bool useCaptureDevice);

class Feature {

private:
  Ptr<FeatureDetector> detector;
  Ptr<DescriptorExtractor> extractor;

public:
  Feature() {

    // SIFT,SURFを使用する場合は初期化
    initModule_nonfree();

    detector = FeatureDetector::create(FEATURE_DETECTOR_TYPE);

    if (FEATURE_DETECTOR_TYPE == "SIFT") {
      detector->set("nFeatures", 0);              // The number of best features to retain
      detector->set("nOctaveLayers", 3);          // The number of layers in each octave
      detector->set("contrastThreshold", 0.001);  // Used to filter out weak features
      detector->set("edgeThreshold", 10);         // Used to filter out edge-like features
      detector->set("sigma", 1.6);                // The sigma of the Gaussian
    }
    else if (FEATURE_DETECTOR_TYPE == "Dense") {
      detector->set("initFeatureScale", 8.f);          // 2*Scale = 20～30が最適
      detector->set("featureScaleLevels", 3);          // Generates several levels of features
      detector->set("featureScaleMul", 1.3f);          // The level parameters are multiplied by
      detector->set("initXyStep", 10);                 // Grid size
      detector->set("initImgBound", 0);                // Excludes the image boundary
      detector->set("varyXyStepWithScale", false);     // The grid node size is multiplied
      detector->set("varyImgBoundWithScale", false);   // Size of image boundary is multiplied
    }
    else if (FEATURE_DETECTOR_TYPE == "GridSIFT") {
      detector->set("maxTotalKeypoints", 2000);    // Maximum count of keypoints detected on the image
      detector->set("gridRows", 8);                // Grid row count
      detector->set("gridCols", 8);                // Grid col count
    }

    extractor = DescriptorExtractor::create(DESCRIPTOR_EXTRACTOR_TYPE);

  }

  void detectKeypoints(Mat image, vector<KeyPoint> &keypoints) {

    detector->detect(image, keypoints);
    if (keypoints.size() < 1) {
#ifdef _DEBUG
      cout << "Warning: No keypoint is detected." << endl;
#endif
    }

  }

  void detectKeypointsBGR(Mat image, vector<vector<KeyPoint>> &keypointsBGR) {

    vector<Mat> imageBGR(3);
    split(image, imageBGR);

    for (int i = 0; i < 3; i++) {
      detector->detect(imageBGR.at(i), keypointsBGR.at(i));
      if (keypointsBGR.at(i).size() < 1) {
#ifdef _DEBUG
        cout << "Warning: No keypoint is detected." << endl;
#endif
      }
    }

  }

  void detectKeypointsWithMask(Mat image, Mat mask, vector<KeyPoint> &keypoints) {

    detector->detect(image, keypoints, mask);
    if (keypoints.size() < 1) {
#ifdef _DEBUG
      cout << "Warning: No keypoint is detected." << endl;
#endif
    }

  }

  void detectKeypointsBGRWithMask(Mat image, Mat mask, vector<vector<KeyPoint>> &keypointsBGR) {

    vector<Mat> imageBGR(3);
    split(image, imageBGR);

    for (int i = 0; i < 3; i++) {
      detector->detect(imageBGR.at(i), keypointsBGR.at(i), mask);
      if (keypointsBGR.at(i).size() < 1) {
#ifdef _DEBUG
        cout << "Warning: No keypoint is detected." << endl;
#endif
      }
    }

  }

  void detectAndDescript(Mat image, vector<KeyPoint> &keypoints, Mat &descriptors) {

    detector->detect(image, keypoints);
    if (keypoints.size() > 0) {
      extractor->compute(image, keypoints, descriptors);
    }
    else {
#ifdef _DEBUG
      cout << "Warning: No keypoint is detected." << endl;
#endif
    }

  }

  void detectAndDescriptBGR(Mat image, vector<vector<KeyPoint>> &keypointsBGR, vector<Mat> &descriptorsBGR){

    vector<Mat> imageBGR(3);
    split(image, imageBGR);

    for (int i = 0; i < 3; i++) {
      detector->detect(imageBGR.at(i), keypointsBGR.at(i));
    }

    for (int i = 0; i < 3; i++) {
      if (keypointsBGR.at(i).size() > 0) {
        extractor->compute(imageBGR.at(i), keypointsBGR.at(i), descriptorsBGR.at(i));
      }
      else {
#ifdef _DEBUG
        cout << "Warning: No keypoint is detected." << endl;
#endif
      }
    }

  }

  void drawKeypointsInCircle(Mat &frame, vector<KeyPoint> &keypoints) {

    for (vector<KeyPoint>::iterator it = keypoints.begin(); it != keypoints.end(); ++it)  {
      double arrow_y, arrow_x;
      arrow_x = (it->size) * cos(it->angle * M_PI / 180.0);
      arrow_y = (it->size) * sin(it->angle * M_PI / 180.0);
      circle(
        frame,
        Point2d(it->pt.x, it->pt.y),
        saturate_cast<int> (it->size),
        Scalar(0, 0, 255),
        1,
        CV_AA
        );
      line(
        frame,
        Point2d(it->pt.x, it->pt.y),
        Point2d(it->pt.x + arrow_x, it->pt.y + arrow_y),
        Scalar(0, 0, 255),
        1,
        CV_AA
        );
    }
  }

  void drawKeypointsInSquare(Mat &frame, vector<KeyPoint> &keypoints) {

    for (vector<KeyPoint>::iterator it = keypoints.begin(); it != keypoints.end(); ++it)  {

      double arrow_y, arrow_x;
      arrow_x = (it->size) * cos(it->angle * M_PI / 180.0);
      arrow_y = (it->size) * sin(it->angle * M_PI / 180.0);

      RotatedRect rotRect = RotatedRect(Point2f(it->pt.x, it->pt.y), Size2f(2 * it->size, 2 * it->size), it->angle);
      Point2f vertices[4];
      rotRect.points(vertices);
      for (int i = 0; i < 4; i++) {
        line(frame, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 255), 1, CV_AA);
      }

      /*line(
      frame,
      Point2d(it->pt.x, it->pt.y),
      Point2d(it->pt.x + arrow_x, it->pt.y + arrow_y),
      Scalar(0,0,255),
      1,
      CV_AA
      );*/

    }
  }

};
class BOW {

private:
  Ptr<BOWKMeansTrainer> bowTrainer;
  Ptr<BOWKMeansTrainer> bowTrainerB;
  Ptr<BOWKMeansTrainer> bowTrainerG;
  Ptr<BOWKMeansTrainer> bowTrainerR;
  Ptr<DescriptorExtractor> dExtractor;
  Ptr<DescriptorMatcher> dMatcher;
  Ptr<BOWImgDescriptorExtractor> bowDExtractor;
  Ptr<BOWImgDescriptorExtractor> bowDExtractorB;
  Ptr<BOWImgDescriptorExtractor> bowDExtractorG;
  Ptr<BOWImgDescriptorExtractor> bowDExtractorR;

public:
  BOW() {

    dExtractor = DescriptorExtractor::create(DESCRIPTOR_EXTRACTOR_TYPE);
    dMatcher = DescriptorMatcher::create("FlannBased");
    /*    ※対応点検索のアルゴリズム(実数ベクトルの場合):
    BruteForce(uses L2), BruteForce-L1, FlannBased    */

  }

  int setDExtractor(string codebookFilename) {


    bowDExtractor = Ptr<BOWImgDescriptorExtractor>(new BOWImgDescriptorExtractor(dExtractor, dMatcher));

    // コードブックの読み込み
    Mat codebook;
    FileStorage cvFileStorageRead(codebookFilename, FileStorage::READ);
    if (!cvFileStorageRead.isOpened()) {
      cerr << "Error: Could not open the codebook." << endl;
      return -1;
    }
    cvFileStorageRead["Codebook"] >> codebook;
    if (codebook.empty()) {
      cerr << "Error: Could not load the codebook." << endl;
      return -1;
    }
    cout << "Loaded " << codebookFilename << endl;

    bowDExtractor->setVocabulary(codebook);

    return 0;

  }

  int setDExtractorBGR(string codebookFilename) {

    bowDExtractorB = Ptr<BOWImgDescriptorExtractor>(new BOWImgDescriptorExtractor(dExtractor, dMatcher));
    bowDExtractorG = Ptr<BOWImgDescriptorExtractor>(new BOWImgDescriptorExtractor(dExtractor, dMatcher));
    bowDExtractorR = Ptr<BOWImgDescriptorExtractor>(new BOWImgDescriptorExtractor(dExtractor, dMatcher));

    // コードブックの読み込み
    vector<Mat> codebook(3);
    FileStorage cvFileStorageRead(codebookFilename, FileStorage::READ);
    if (!cvFileStorageRead.isOpened()) {
      cerr << "Error: Could not open the codebook." << endl;
      return -1;
    }

    cvFileStorageRead["CodebookB"] >> codebook.at(0);
    cvFileStorageRead["CodebookG"] >> codebook.at(1);
    cvFileStorageRead["CodebookR"] >> codebook.at(2);

    for (int i = 0; i < 3; i++) {
      if (codebook.at(i).empty()) {
        cerr << "Error: Could not load the codebook." << endl;
        return -1;
      }
    }
    cout << "Loaded " << codebookFilename << endl;

    bowDExtractorB->setVocabulary(codebook.at(0));
    bowDExtractorG->setVocabulary(codebook.at(1));
    bowDExtractorR->setVocabulary(codebook.at(2));

    return 0;

  }

  int descript(Mat image, vector<KeyPoint> &keypoints, Mat &bowDescriptors){

    if (keypoints.size() > 0) {
      bowDExtractor->compute(image, keypoints, bowDescriptors);
    }
    else {
#ifdef _DEBUG
      cout << "Warning: No keypoint is detected." << endl;
#endif
      return -1;
    }
    return 0;

  }

  int descriptBGR(Mat image, vector<vector<KeyPoint>> &keypointsBGR, vector<Mat> &bowDescriptorsBGR, Mat &bowDescriptors){

    vector<Mat> imageBGR(3);
    split(image, imageBGR);

    vector<Mat> descriptorsBGR(3);

    if (keypointsBGR.at(0).size() > 0) {
      bowDExtractorB->compute(imageBGR.at(0), keypointsBGR.at(0), descriptorsBGR.at(0));
      bowDescriptors = descriptorsBGR.at(0);
    }
    else {
#ifdef _DEBUG
      cout << "Warning: No keypoint is detected in blue channel." << endl;
#endif
      return -1;
    }

    if (keypointsBGR.at(1).size() > 0) {
      bowDExtractorG->compute(imageBGR.at(1), keypointsBGR.at(1), descriptorsBGR.at(1));
      hconcat(bowDescriptors, descriptorsBGR.at(1), bowDescriptors);
    }
    else {
#ifdef _DEBUG
      cout << "Warning: No keypoint is detected in green channel." << endl;
#endif
      return -1;
    }

    if (keypointsBGR.at(2).size() > 0) {
      bowDExtractorR->compute(imageBGR.at(2), keypointsBGR.at(2), descriptorsBGR.at(2));
      hconcat(bowDescriptors, descriptorsBGR.at(2), bowDescriptors);
    }
    else {
#ifdef _DEBUG
      cout << "Warning: No keypoint is detected in red channel." << endl;
#endif
      return -1;
    }

    /*cout << descriptorsBGR.at(0) << endl;
    cout << descriptorsBGR.at(1) << endl;
    cout << descriptorsBGR.at(2) << endl;
    cout << bowDescriptors << endl;*/

    return 0;

  }

  void setBOWTrainer(){

    const int clusterCount = VISUAL_WORDS_COUNT;    // VISUAL_WORDS_COUNT個にクラスタリング
    const TermCriteria& termcrit = TermCriteria();  // 反復数の最大値/精度
    const int attempts = 5;                         // 試行回数
    const int flags = KMEANS_PP_CENTERS;            // RANDOM_CENTERS, PP_CENTERS, USE_INITIAL_LABELS
    bowTrainer = Ptr<BOWKMeansTrainer>(new BOWKMeansTrainer(clusterCount, termcrit, attempts, flags));

  }

  void setBOWTrainerBGR(){

    const int clusterCount = VISUAL_WORDS_COUNT;    // VISUAL_WORDS_COUNT個にクラスタリング
    const TermCriteria& termcrit = TermCriteria();  // 反復数の最大値/精度
    const int attempts = 5;                         // 試行回数
    const int flags = KMEANS_PP_CENTERS;            // RANDOM_CENTERS, PP_CENTERS, USE_INITIAL_LABELS
    bowTrainerB = Ptr<BOWKMeansTrainer>(new BOWKMeansTrainer(clusterCount, termcrit, attempts, flags));
    bowTrainerG = Ptr<BOWKMeansTrainer>(new BOWKMeansTrainer(clusterCount, termcrit, attempts, flags));
    bowTrainerR = Ptr<BOWKMeansTrainer>(new BOWKMeansTrainer(clusterCount, termcrit, attempts, flags));

  }

  void addToBOWTrainer(Mat descriptors){

    bowTrainer->add(descriptors);

  }

  void addToBOWTrainerBGR(vector<Mat> descriptorsBGR){

    bowTrainerB->add(descriptorsBGR.at(0));
    bowTrainerG->add(descriptorsBGR.at(1));
    bowTrainerR->add(descriptorsBGR.at(2));

  }

  int createCodebook(string codebookFilename, int keypointTotal){

    if (bowTrainer->descripotorsCount() == 0) {
      cerr << "Error: Could not load the images." << endl;
      return -1;
    }

    cout << "Keypoint(total): " << keypointTotal << endl;
    cout << "Training visual vocabulary (may take a few hours or days)..." << endl;

    Mat codebook = bowTrainer->cluster();
    FileStorage cvFileStorage(codebookFilename, FileStorage::WRITE);
    write(cvFileStorage, "KeypointTotal", keypointTotal);
    write(cvFileStorage, "Codebook", codebook);
    cout << "Saved the bow codebook." << endl;

    return 0;

  }

  int createCodebookBGR(string codebookFilename, vector<int> keypointTotal){

    if (bowTrainerB->descripotorsCount() == 0) {
      cerr << "Error: Could not load the images." << endl;
      return -1;
    }

    FileStorage cvFileStorage(codebookFilename, FileStorage::WRITE);
    write(cvFileStorage, "KeypointTotal", keypointTotal);
    cout << "Keypoint(total): " << keypointTotal.at(0) << " "
      << keypointTotal.at(1) << " " << keypointTotal.at(2) << endl;

    cout << "Training visual vocabulary (may take a few hours or days)..." << endl;
    Mat codebookB = bowTrainerB->cluster();
    write(cvFileStorage, "CodebookB", codebookB);
    cout << "Saved the bow codebook of blue channel." << endl;
    Mat codebookG = bowTrainerG->cluster();
    write(cvFileStorage, "CodebookG", codebookG);
    cout << "Saved the bow codebook of green channel." << endl;
    Mat codebookR = bowTrainerR->cluster();
    write(cvFileStorage, "CodebookR", codebookR);
    cout << "Saved the bow codebook of red channel." << endl;

    return 0;

  }

};
class ClassifierSVM {

private:
  SVMParams svmParams;
  SVM svm;
  const int kFold;

public:
  ClassifierSVM() : kFold(10) {}
  void build(Mat &features, Mat &responses, string ClassifierFilename) {

    // SVMのパラメータを設定
    svmParams.svm_type = SVM::C_SVC;    // C_SVC, NU_SVC, ONE_CLASS, EPS_SVR, NU_SVR
    svmParams.kernel_type = SVM::RBF;   // LINEAR, POLY, RBF, SIGMOID
    svmParams.C = 100;

    svm.train(features, responses, Mat(), Mat(), svmParams);
    svm.save(ClassifierFilename.c_str());

  }

  int load(string classifierFilename) {

    svm.load(classifierFilename.c_str());
    if (svm.get_var_count() == 0) {
      cerr << "Error: Could not open the classifier." << endl;
      return -1;
    }
    cout << "Loaded " << classifierFilename << endl;
    return 0;

  }

  void predict(Mat &features, Mat &results) {

    svm.predict(features, results);

  }

};
class Dataset {

public:
  vector<string> filenames;
  Mat features;   // 特徴量
  Mat responses;  // クラス

};
