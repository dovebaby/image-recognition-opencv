#include "image_recog.h"

int main(int argc, char *argv[]) {

  // ファイル名設定
  string codebookFilename = "K=" + to_string(VISUAL_WORDS_COUNT) + "_codebook.xml";
  string testDataFilename = "K=" + to_string(VISUAL_WORDS_COUNT) + "_train_data.xml";
  string trainDataFilename = "K=" + to_string(VISUAL_WORDS_COUNT) + "_test_data.xml";
  string classifierFilename = "K=" + to_string(VISUAL_WORDS_COUNT) + "_svm.xml";

  /*createBOWCodebook(codebookFilename, "./Train", 50);
  convertImageToBOW(codebookFilename, "./Train", 100, trainDataFilename);
  convertImageToBOW(codebookFilename, "./Test", 100, testDataFilename);
  trainClassifier(trainDataFilename, classifierFilename, "train_results.txt");
  testClassifier(testDataFilename, classifierFilename, "test_results.txt");*/

  Mat image;
  //image = imread(argv[1], 1);   // ドラッグ&ドロップ
  image = imread("test.jpg", 1);  // ファイル名指定
  if (image.empty()) {
    cerr << "Error: Could not open the image." << endl;
    return -1;
  }
  recognizeImage(codebookFilename, trainDataFilename, classifierFilename, image, false);
  //recognizeImage(codebookFilename, trainDataFilename, classifierFilename, Mat(), true);

  return 0;
}
