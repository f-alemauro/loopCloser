#include "ros/ros.h"
#include "Vocabulary.h"
#include <stdexcept>

int main(int argc, char **argv)
{
	string detectorType, descriptorTpye, clusteringMethod, datasetDirectory, fileExtension;
	bool truncate;
	float truncateFirst, truncateLast;
	int numberOfWords;
	unsigned long featuresNumber, descriptorNumber, clusterNumber;

	if(argc!=10 && argc!=8){
		ROS_ERROR("Improper use of function parameters!");
		ROS_ERROR("Usage: ./generateVocab detectorType descriptorType clusteringMethods datasetFolder imageExtension numerOfWords useTruncate [truncateFirstValue TruncateLastValue]");
		return -1;
	}


	detectorType = argv[1];
	descriptorTpye = argv[2];
	clusteringMethod = argv[3];
	datasetDirectory = argv[4];
	fileExtension = argv[5];
	numberOfWords = atoi(argv[6]);
	truncate = atoi(argv[7]);
	if(argc==10)
	{
		truncateFirst= atof(argv[8]);
		truncateLast= atof(argv[9]);
	}
	ofstream myfile;
	myfile.open ("outData/param/generateVocab.txt");
	myfile << argv[0]<<" "<< detectorType << " " << descriptorTpye << " "<<clusteringMethod << " "<<datasetDirectory <<" "<< fileExtension <<" "<< numberOfWords <<" "<< truncate <<" "<< truncateFirst <<" "<< truncateLast;
	myfile.close();

	Vocabulary VocabSet;
	try{
		VocabSet.acquireImgNames(datasetDirectory, fileExtension);
	}catch(std::string& e){
		ROS_ERROR("%s",e.c_str());
		return -1;
	}
	try{
		featuresNumber = VocabSet.extractFeatures_TS(detectorType, descriptorTpye);
	}catch(std::runtime_error & e){
		ROS_ERROR("%s",e.what());
		return -1;
	}
	if (featuresNumber!=0){
		try{
			descriptorNumber = VocabSet.describeFeatures_TS(descriptorTpye);
		}catch(std::runtime_error & e){
			ROS_ERROR("%s",e.what());
			return -1;
		}
		if (descriptorNumber!=0){
			try{
				clusterNumber = VocabSet.clusterDescriptors(clusteringMethod, numberOfWords);
			}catch(std::runtime_error & e) {
				ROS_ERROR("%s",e.what());
				return -1;
			}
			if (clusterNumber != 0){
				if (truncate){
					try {
						VocabSet.buildAllIndeces("general_index.ind","general_inverted_index.ind");
						VocabSet.truncateVocabulary(truncateFirst, truncateLast);
					} catch (std::runtime_error & e) {
						ROS_ERROR("%s",e.what());
						return -1;
					}

				}
			}
			else
			{
				ROS_ERROR("No cluster generated!");
				return -1;
			}

		}
		else
		{
			ROS_ERROR("No descriptor generated!");
			return -1;
		}
	}
	else{
		ROS_ERROR("No feature extracted!");
		return -1;
	}
	return 0;
}


