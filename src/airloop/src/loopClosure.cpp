#include "ros/ros.h"
#include "cv_bridge/cv_bridge.h"
#include "Vocabulary.h"
#include "Loop_Closer.h"
#include "boost/filesystem.hpp"
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Int32MultiArray.h>
#include <std_msgs/Int8.h>

const int defaultNolcWith = 50;
const float defaultProbabThreshold = 0.3;
const int defaultNumbNeigh = 5;
const float defaultGeomThreshold = 0.4;

Vocabulary VocabSet;
Loop_Closer ClosureDetection;
string VocabularyAddress;
ros::Publisher img_pub;

unsigned int img_n;
int NOLC_WITH;
float PROBAB_THRESH;
int N_NEIGH;
float GEOM_THRESH;
int lc;

using namespace boost::filesystem;
using namespace cv;

void odomProcessing(const std_msgs::Int8 msg){
	ROS_INFO("YEEEEE: %d",msg.data);
}


void imageProcessing(const sensor_msgs::ImageConstPtr& msg, flann::GenericIndex<cv::flann::L2<float> > &flannIndObj){
	std::vector<std::vector<float> > lc;
	std_msgs::Int32MultiArray imgs;
	ROS_INFO("New image found!");
	cv_bridge::CvImagePtr cv_ptr;
	try
	{
		cv_ptr = cv_bridge::toCvCopy(msg);
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("Error in converting image to openCV format: %s",e.what());
		return;
	}
	try{
		lc = ClosureDetection.calcProbabCamera(VocabSet, flannIndObj, NOLC_WITH, PROBAB_THRESH, N_NEIGH, GEOM_THRESH, cv_ptr->image, img_n);
	}
	catch (runtime_error &e)
	{
		cout<<e.what()<<endl;
	}
	img_n++;
	for(int i=0;i<lc.size();i++){
		imgs.data.push_back(lc[i][0]);
		imgs.data.push_back(lc[i][1]);
	}
	if(lc.size()!=0 && ros::ok() && img_pub.getNumSubscribers()>0)
		img_pub.publish(imgs);
	else if (img_pub.getNumSubscribers()==0)
		ROS_ERROR("No active subscriber for loopClosing!");
	else if(!ros::ok())
		ROS_ERROR("Error in ROS");
}

int main(int argc, char **argv)
{
	if(argc < 2){
		ROS_ERROR("Missing dictionary directory!\nUsage: loopClosure dictionaryDirectory [numberOfSkippedFrame loopClosureThresh directNeighboors geomThresh]");
		return -1;
	}
	else if(argc != 6 ){
		ROS_INFO("Wrong number of parameters!\nUsage: loopClosure dictionaryDirectory [numberOfSkippedFrame loopClosureThresh directNeighboors geomThresh]\nAssuming default parameters value.");
		NOLC_WITH = defaultNolcWith;
		PROBAB_THRESH = defaultProbabThreshold;
		N_NEIGH = defaultNumbNeigh;
		GEOM_THRESH = defaultGeomThreshold;
		ROS_INFO("numberOfSkippedFrame: %d\nloopClosureThresh: %.3f\ndirectNeighboors: %d\ngeomThresh: %.3f",defaultNolcWith,defaultProbabThreshold,defaultNumbNeigh,defaultGeomThreshold);
	}
	else{
		NOLC_WITH = atoi(argv[2]);
		PROBAB_THRESH = atoi(argv[3]);
		N_NEIGH = atoi(argv[4]);
		GEOM_THRESH = atoi(argv[5]);

	}

	if(is_regular_file(argv[1]))
		VocabularyAddress=argv[1];
	else
	{
		ROS_ERROR("The dictionary file does not exists!");
		return -1;
	}

	img_n=0;
	ROS_INFO("Reading vocabulary...");
	unsigned long numberOfEntries = VocabSet.readDict(VocabularyAddress);
	ROS_INFO("Read %lu from vocabulary.", numberOfEntries);

	cv::Mat Vocabulary = VocabSet.takeDictionary();

	cvflann::KMeansIndexParams flannIndexParams(32, 11, cvflann::FLANN_CENTERS_RANDOM, 0.2);
	flann::GenericIndex<cv::flann::L2<float> > FlannObj(Vocabulary,(cvflann::AutotunedIndexParams&)flannIndexParams);
	ROS_INFO("Flann Index successfully generated");
	ROS_INFO("Init loopClosure node...");

	ofstream myfile;
	myfile.open ("outData/loopClosure_params.txt");

	myfile << argv[0]<<" "<< VocabularyAddress << " " << NOLC_WITH << " "<<PROBAB_THRESH << " "<<N_NEIGH <<" "<< GEOM_THRESH;
	myfile.close();

	ros::init(argc, argv, "loopClosure");

	ros::NodeHandle n;
	ros::Subscriber img_sub = n.subscribe<sensor_msgs::Image>("/images", 100, boost::bind(imageProcessing, _1, boost::ref(FlannObj)));
	ros::Subscriber odom_sub = n.subscribe<std_msgs::Int8>("/odom", 100, odomProcessing);
	img_pub = n.advertise<std_msgs::Int32MultiArray>("/loopClosures", 100);
	ros::spin();
	return 0;
}


