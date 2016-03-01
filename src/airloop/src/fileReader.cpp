#include "ros/ros.h"
#include "std_msgs/Time.h"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include "boost/filesystem.hpp"
#include <boost/foreach.hpp>
#include <fstream>

using namespace boost::filesystem;
using namespace std;
using namespace cv;
using namespace ros;

ros::Time ts;

int get_all(const path& root, const string& ext, vector<path>& ret)
{
	int count = 0;
	if(!exists(root) || !is_directory(root) || is_empty(root))
		return -1;
	recursive_directory_iterator it(root);
	recursive_directory_iterator endit;
	while(it != endit)
	{
		if(is_regular_file(*it) && it->path().extension() == ext){
			count++;
			ret.push_back(it->path());
		}
		++it;
	}
	return count;
}

void newTs(const std_msgs::Time time){
	ROS_INFO("new ts read");
	ts = time.data;
}

int main(int argc, char **argv)
{
	int loopRate;
	vector<path> imageNames;
	vector<path>::const_iterator iter;
	cv_bridge::CvImage cv_image;
	sensor_msgs::Image ros_image;
	if(argc < 2){
		ROS_ERROR("No path specified! Correct usage is ./fileReader directory fileExtension");
		return -1;
	}
	if(argc < 3){
		ROS_ERROR("No extension specified! Correct usage is ./fileReader directory fileExtension");
		return -1;
	}
	if(argc < 4){
		ROS_ERROR("Missing loop rate! Setting default value: 5");
		loopRate = 5;
	}
	else
		loopRate = atoi(argv[3]);


	ofstream myfile;
	myfile.open ("outData/param/fileReader.txt",std::ofstream::app);
	for (int i = 0; i< argc;i++)
		myfile << argv[i]<<" ";
	myfile<<"\n";
	myfile.close();

	ros::init(argc, argv, "fileReader");
	NodeHandle n;
	Publisher img_pub = n.advertise<sensor_msgs::Image>("/images", 100);
	Subscriber ts_sub = n.subscribe<std_msgs::Time>("/timeStamp", 1, newTs);
	if(img_pub){
		Rate loop_rate(loopRate);
		int result = get_all(argv[1], argv[2], imageNames);
		if(result == -1){
			ROS_ERROR("Directory does not exist!");
			return -1;
		}
		if(result == 0){
			ROS_ERROR("No %s file found!", argv[2]);
			return -1;
		}
		sort(imageNames.begin(),imageNames.end());
		while(img_pub.getNumSubscribers()<1 && ok())
			ROS_INFO("Waiting for subscribers...");

		for(iter = imageNames.begin(); iter!= imageNames.end()&& ok() && img_pub.getNumSubscribers()>0;++iter){
			spinOnce();
			loop_rate.sleep();
			ROS_INFO(iter->generic_string().c_str());
			cv_image.image = imread(iter->generic_string(),CV_LOAD_IMAGE_COLOR);
			cv_image.encoding = "bgr8";
			cv_image.toImageMsg(ros_image);
			ros_image.header.stamp = ts;
			img_pub.publish(ros_image);

		}
		if (img_pub.getNumSubscribers()==0){
			ROS_ERROR("No more subscriber! Quitting!");
			return 0;
		}
		else if(!ok()){
			ROS_ERROR("Error in ROS! Quitting!");
			return -1;
		}
	}
	else{
		ROS_ERROR("Error in creating publisher!");
		return -1;
	}
	return 0;
}
