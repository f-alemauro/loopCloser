#ifndef VOCABULARY_H_
#define VOCABULARY_H_

#include "/usr/local/include/opencv2/highgui/highgui.hpp"
#include "/usr/local/include/opencv2/imgproc/imgproc.hpp"
#include "/usr/local/include/opencv2/nonfree/features2d.hpp"
#include "/usr/local/include/opencv2/flann/flann.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include <utility>
#include <deque>
#include <vector>
#include <map>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/param.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>

#include "boost/filesystem.hpp"

using namespace std;
using namespace cv;

class Vocabulary {
public:
    explicit Vocabulary(std::string base_inp="Images"):
                                 imgDirectory (base_inp), TotalImagesNumber(0),
                                 TotalDescriptorNumber(0), dataDimension(0) { }
    void acquireImgNames (const std::string &dir_path, const std::string &fileExtension) throw(std::string);
    unsigned long extractFeatures_TS  (const std::string &DetectorType, const std::string &DescriptorType) throw(std::runtime_error);
    unsigned long describeFeatures_TS (const std::string &DescriptorType)  throw(std::runtime_error);
    unsigned long clusterDescriptors(const string &method,const int &number_cluster) throw(std::runtime_error);
    void buildAllIndeces (const string& Index_path, const string& InvIndex_path) throw (runtime_error);
    void truncateVocabulary(const float upperLimit, const float lowerLimit) throw (runtime_error);
    unsigned long readDict(const std::string &treadIndexDictionary) throw (runtime_error);
    void readFabMapDict(const std::string &treadIndexDictionary);                               //+
    cv::Mat takeDictionary(){return ClusterCenters;}
    std::vector<int> getSowBowRepr_img (const cv::Mat &current_picture, const int &method, cv::flann::GenericIndex<cv::flann::L2<float> >  &flannIndObj,std::vector<cv::KeyPoint> &curImgFeatures, cv::Mat &curImgDescriptors, cv::Mat &flann_labels,     cv::Mat &flann_dist ) const;  // <--- overloading  is possible
    std::vector<int> getSowBowRepr_img (cv::Mat &imgDescr, cv::flann::GenericIndex<cv::flann::L2<float> >  &flannIndObj,  cv::Mat &flann_labels, cv::Mat &flann_dist  )  const;

private:
    inline std::string generatePath (const std::string &inp_base, const std::size_t &number, const std::string &inp_extension) const;           // generate address of the next file to write to
    std::vector<cv::KeyPoint> getImgFeatures(const cv::Mat &img_gray, const cv::Ptr<cv::FeatureDetector> &detector)const;
    cv::Mat describeImgFeatures (const cv::Mat &cur_image, std::vector<cv::KeyPoint> &key_points, const cv::Ptr<cv::DescriptorExtractor> &descriptorExtractor)const;
    std::vector <std::string> listFilesInDirectory (const std::string &dir_path, const std::string &ext) const;
    cv::Mat grabImage(const string &path, const int &grayscale) const;
    void MatToFile (const std::string&, cv::Mat) const;																							 
    inline void generateFolders (const std::string &kp_folder, const std::string &descr_folder, const std::string &outdata) const;				 
    std::vector<int> getFlannSowBowVector (flann::GenericIndex< cvflann::L2< float > >& flannIndex, Mat& descr_query, Mat& indices, Mat& dist, int knn, cvflann::SearchParams params, int sow_bow) const;
    cv::Mat describeImgSurfFeatures (const cv::Mat &img_gray, std::vector<cv::KeyPoint> &key_points,const int number1, const int number2, const bool number3 )const;
    cv::Mat describeImgBriefFeatures (const cv::Mat &img_gray, std::vector<cv::KeyPoint> &key_points,const int bits_n )const;

    std::map<int, int> vw_occurrence_n;
    std::map<int, std::map<int,int> > sowIndex;
    std::map<int, std::vector<int> > sowInvertIndex;

    std::string imgDirectory;
    std::size_t TotalImagesNumber;
    std::size_t TotalDescriptorNumber;

    cv::Mat idf_voc;                                    // calc. when invInd is built or read

    std::string DescriptorExtractorType;
    std::string FeatureDetectorType;
    unsigned dataDimension;

    cv::Mat PointToCluster,ClusterCenters;
    std::deque <std::vector<cv::KeyPoint> > KeyPointsByImage;
    std::deque <cv::Mat> AllDescriptorsByImage;
    cv::Mat GreatMatrix;								//temporary made private attribute

    std::vector <std::string> ImgNames;
    std::string outputFolder = "outData/";
    std::string kp_folder = "KeyPoints";
    std::string descr_folder ="Descriptors";;
    std::string outdata = "GenData/";
};

#endif /* VOCABULARYGENERATOR_H_ */
