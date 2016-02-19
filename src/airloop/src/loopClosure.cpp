#include "ros/ros.h"
#include "cv_bridge/cv_bridge.h"
#include "Vocabulary.h"
#include "Loop_Closer.h"
#include "boost/filesystem.hpp"
#include <getopt.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Int32MultiArray.h>
#include <std_msgs/Int8.h>
#include "sensor_msgs/Imu.h"

const int defaultNolcWith = 50;
const float defaultProbabThreshold = 0.3;
const int defaultNumbNeigh = 5;
const float defaultGeomThreshold = 0.4;
const bool defaultLoadDB = false;

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
bool LOADDB;


using namespace boost::filesystem;
using namespace cv;

//void odomProcessing(const sensor_msgs::Imu msg){
//	ROS_INFO("YEEEEE: %d",msg.header.stamp.sec);
//}


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

	int ts = (*msg).header.stamp.sec;
	int nts = (*msg).header.stamp.nsec;
	int ms = nts/1000000;
	int sec = ts%60;
	int min = (ts/60)%60;
	int ora = (ts/3600)%24;
	stringstream temp;
	temp << ora << "_"<< min << "_"<< sec << "_"<< ms<<".ppm";

	try{
		lc = ClosureDetection.calcProbabCamera(VocabSet, flannIndObj, NOLC_WITH, PROBAB_THRESH, N_NEIGH, GEOM_THRESH, cv_ptr->image, img_n, temp.str().c_str());
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

void printUsage(){
	cout<<"**********"<<endl;
	cout<<"Usage of loopClosure:"<<endl<<endl;
	cout<<"./loopClosure"<<endl;
	cout<<"--dictionaryDir=value, mandatory argument, specify the dictionary file."<<endl<<endl;
	cout<<"--loadDBDir=value, optional parameter with mandatory argument, specify the database directory you wish to load. (Not compatible with option --newDBName)"<<endl<<endl;
	cout<<"--newDBName=value, optional parameter with mandatory argument, specify the name of the new database. (Not compatible with option --loadDBDir)"<<endl<<endl;
	cout<<"--numberOfSkippedFrame=value, optional parameter with mandatory argument, specify the number of frame to be skipped before performing loop closing."<<endl<<endl;
	cout<<"--loopClosureThresh=value, optional parameter with mandatory argument, specify the threshold for loop closing."<<endl<<endl;
	cout<<"--directNeighboors=value, optional parameter with mandatory argument, specify the number of neighbor frames."<<endl<<endl;
	cout<<"--geomThresh=value, optional parameter with mandatory argument, specify the geometry threshold for loop closing."<<endl<<endl;
	cout<<"--help, optional parameter without argument, print this help."<<endl<<endl;
	cout<<"**********"<<endl;
}


int main(int argc, char **argv)
{
	string dbAddress, newDbName;
	int aFlag=0,bFlag=0,cFlag=0,dFlag=0,eFlag=0,fFlag=0,gFlag=0;

	NOLC_WITH = defaultNolcWith;
	PROBAB_THRESH = defaultProbabThreshold;
	N_NEIGH = defaultNumbNeigh;
	GEOM_THRESH = defaultGeomThreshold;
	LOADDB = defaultLoadDB;

	int ch;
	int option_index;
	struct option longopts[] = {
			{"dictionaryDir", required_argument, 0, 'a'},
			{"loadDBDir", required_argument, 0, 'b'},
			{"newDBName",required_argument, 0, 'c'},
			{"numberOfSkippedFrame",required_argument, 0, 'd'},
			{"loopClosureThresh",required_argument, 0, 'e'},
			{"directNeighboors",required_argument, 0, 'f'},
			{"geomThresh", required_argument, 0, 'g'},
			{"help", no_argument, 0, 'h'},
	};
	while ((ch = getopt_long(argc, argv, ":a:b:c:d:e:f:g:h", longopts, &option_index)) != -1) {
		switch (ch) {
		case 'a'://dictionaryDiroption_index
			if(is_regular_file(optarg))
				VocabularyAddress=optarg;
			else
			{
				ROS_ERROR("The dictionary file does not exist!");
				return -1;
			}
			aFlag =1;
			break;
		case 'b'://loadDBDir
			LOADDB = true;
			if(exists(optarg)){
				dbAddress=optarg;
			}else
			{
				ROS_ERROR("The db directory does not exist!");
				return -1;
			}
			bFlag =1;
			break;
		case 'c'://newDBName
			newDbName = optarg;
			cFlag =1;
			break;
		case 'd'://numberOfSkippedFrame
			NOLC_WITH = atoi(optarg);
			dFlag = 1;
			break;
		case 'e'://loopClosureThresh
			PROBAB_THRESH = atoi(optarg);
			eFlag = 1;
			break;
		case 'f'://directNeighboors
			N_NEIGH = atoi(optarg);
			fFlag = 1;
			break;
		case 'g'://geomThresh
			GEOM_THRESH = atoi(optarg);
			gFlag = 1;
			break;
		case 'h'://help
			printUsage();
			return 0;
			break;
		case ':':/* missing value*/
			ROS_ERROR("Missing parameter value");
			break;
		case '?':/* invalid option*/
		default:/* invalid option*/
			ROS_ERROR("Option is not valid");
			break;
		}

	}
	if (aFlag == 0){
		ROS_ERROR("dictionaryDir parameters is mandatory!");
		return -1;
	}
	if (bFlag+cFlag == 0){
		ROS_ERROR("Either loadDBDir or newDBName must is mandatory");
		return -1;
	}
	if (bFlag+cFlag == 2){
		ROS_ERROR("Error: both loadDBDir and newDBName are set");
		return -1;
	}
	if(dFlag == 0)
		ROS_INFO("No numberOfSkippedFrame specified; loaded default value: %d",defaultNolcWith);
	if(eFlag == 0)
		ROS_INFO("No loopClosureThresh specified; loaded default value: %f", defaultProbabThreshold);
	if(fFlag == 0)
		ROS_INFO("No directNeighboors specified; loaded default value: %d",defaultNumbNeigh);
	if(gFlag == 0)
		ROS_INFO("No geomThresh specified; loaded default value: %f",defaultGeomThreshold);


	img_n=0;
	ROS_INFO("Reading vocabulary...");
	unsigned long numberOfEntries = VocabSet.readDict(VocabularyAddress);
	ROS_INFO("Read %lu from vocabulary.", numberOfEntries);

	cv::Mat Vocabulary = VocabSet.takeDictionary();

	cvflann::KMeansIndexParams flannIndexParams(32, 11, cvflann::FLANN_CENTERS_RANDOM, 0.2);
	flann::GenericIndex<cv::flann::L2<float> > FlannObj(Vocabulary,(cvflann::AutotunedIndexParams&)flannIndexParams);
	ROS_INFO("Flann Index successfully generated");

	if(LOADDB){
		ROS_INFO("Reading db...");
		try{
			img_n = ClosureDetection.readDB(dbAddress);
		}catch (runtime_error &e)
		{
			ROS_ERROR(e.what());
			return -1;
		}
		ROS_INFO("Read %lu db entries", img_n);
	}
	else{
		ROS_INFO("No previously db loaded, new db will be generated");
		ClosureDetection.setupDbDirs(newDbName);
	}


	ROS_INFO("Init loopClosure node...");
	ofstream myfile;
	myfile.open ("outData/param/loopClosure.txt",std::ofstream::app);
	for (int i = 0; i< argc;i++)
		myfile << argv[i]<<" ";
	myfile<<"\n";
	myfile.close();

	ros::init(argc, argv, "loopClosure");

	ros::NodeHandle n;
	ros::Subscriber img_sub = n.subscribe<sensor_msgs::Image>("/images", 100, boost::bind(imageProcessing, _1, boost::ref(FlannObj)));
	//ros::Subscriber odom_sub = n.subscribe<sensor_msgs::Imu>("/odom", 100, odomProcessing);
	img_pub = n.advertise<std_msgs::Int32MultiArray>("/loopClosures", 100);
	ROS_INFO("LoopClosure node correctly created!");
	ros::spin();
	return 0;
}
