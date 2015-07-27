#include "ros/ros.h"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include "boost/filesystem.hpp"
#include <boost/foreach.hpp>
#include <fstream>

using namespace boost::filesystem;
using namespace std;
using namespace cv;
using namespace ros;

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
			cout<<it->path()<<endl;
			ret.push_back(it->path());
		}
		++it;
	}
	return count;
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
	myfile.open ("outData/param/fileReader.txt");
	myfile << argv[0]<<" "<< argv[1] << " " << argv[2] << " " << loopRate;
	myfile.close();

	ros::init(argc, argv, "fileReader");
	NodeHandle n;
	Publisher img_pub = n.advertise<sensor_msgs::Image>("/images", 100);
	cout<<img_pub <<endl;
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

		while(img_pub.getNumSubscribers()<1)
			ROS_INFO("Waiting for subscribers...");

		for(iter = imageNames.begin(); iter!= imageNames.end()&& ok() && img_pub.getNumSubscribers()>0;++iter){
			cout<< iter->generic_string()<<endl;
			cv_image.image = imread(iter->generic_string(),CV_LOAD_IMAGE_COLOR);
			cv_image.encoding = "bgr8";
			if(cv_image.image.empty())
				cout<<"Empty image"<<endl;
			else
				cout<<"Image Ok"<<endl;
			cv_image.toImageMsg(ros_image);
			img_pub.publish(ros_image);
			spinOnce();
			loop_rate.sleep();
		}
		if (img_pub.getNumSubscribers()==0){
			ROS_ERROR("No more subscriber! Quitting!");
			return 0;
		}
		else if(!ok()){
			ROS_ERROR("Error in ROS! Qutting!");
			return -1;
		}
	}
	else{
		ROS_ERROR("Error in creating publisher!");
		return -1;
	}
	return 0;
}
