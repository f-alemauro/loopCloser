#ifndef LOOP_CLOSER_H_
#define LOOP_CLOSER_H_

#include "Vocabulary.h"

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/flann/flann.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <math.h>
#include <string>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <unistd.h>
#include <sys/param.h>
#include <vector>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <numeric>

class Loop_Closer {

public:
	Loop_Closer(){};

	void setupDbDirs(string dbDir){
		outputFolder = dbDir;
		im  = "img/";
		imBow  = "imgBow/";
		imDescr ="imgDescr/";
		imFeat = "imgFeat/";
		imgFlannLab = "imgFlannLab/";
		closureProbabFile ="closProbab.dat";
		vector<std::string> dirs;
		dirs.push_back(imBow);
		dirs.push_back(imDescr);
		dirs.push_back(imFeat);
		dirs.push_back(imgFlannLab);
		dirs.push_back(im);
		generateFolders(dirs);

	}

	std::vector<std::vector<float> > calcProbabCamera(Vocabulary &VocabularyObject,flann::GenericIndex<flann::L2<float> >  &flannIndObj, const unsigned neighbour, const float lc_threshold, const unsigned subset_range, const float gmtr_threshold, Mat &image, unsigned int img_n, string ts) throw (runtime_error);
	unsigned int readDB(string dir) throw (runtime_error);

private:
	int get_all(boost::filesystem::path root, vector<boost::filesystem::path>& ret);
	std::map<int, std::vector<int> > recalcNewInvIndex(    std::vector<int>::iterator curImgBow_beg,std::vector<int>::iterator curImgBow_end, int img_number) const ;
	std::vector<float> calcImg_tfIdf(std::vector<int>::const_iterator bow_beg, std::vector<int>::const_iterator bow_end, const int n_features_img )const;
	std::map<int,float> getSimilarityCoefficients ( std::vector<int>::const_iterator sow_iterat, std::vector<int>::const_iterator sow_iterat_fal, std::vector<float>::const_iterator itr_curImg_w_log, std::vector<float>::const_iterator itr_curImg_w_log_fal, int neighbor, int img_n) const;
	std::vector<int> takeAvgImage(const int vw_n, const int avgImg_rows,  int &n_descr_avgImg)const;
	std::map<int,float> getLikelihoodCoef (const std::map<int,float>::const_iterator sim_beg, const std::map<int,float>::const_iterator sim_end) const;
	cv::Mat calcIdfIset (std::vector<int>::const_iterator bow_beg, const int total_n_vw) const;
	std::map<int, std::vector<int> > newInvIndex;
	std::map<int, int> vw_times_seen;
	std::map<int, std::vector<float> > tfIdfByImage;
	std::map<int, std::vector<float> > closureProbab;
	std::map<int, std::vector<float> > closureProbabSubset;
	std::vector<std::vector<float> > detectedLoopClosures;
	std::vector<std::map<int,float> > lc_probability;
	std::vector<int> bowAvgImage;
	cv::Mat idf;
	std::vector<cv::Mat> newDescriptorsByImage;
	std::vector<std::vector<cv::KeyPoint> > newKeyPointsByImage;
	std::vector<cv::Mat> vwImgRepresByImage;
	std::vector<std::vector<int> > bowImgRepresByImage;

	void generateFolders (vector<std::string> dirs);

	std::string outputFolder;
	std::string im;
	std::string imBow;
	std::string imDescr;
	std::string imFeat;
	std::string imgFlannLab;
	std::string closureProbabFile;
};

#endif /* LOOP_CLOSER_H_ */
