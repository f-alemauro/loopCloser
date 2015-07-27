#include "ros/ros.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/Int32MultiArray.h>
#include "boost/filesystem.hpp"
#include <boost/foreach.hpp>
#include <fstream>

using namespace cv;
using namespace boost::filesystem;


vector<path> imageNames;
vector<Mat> images;

bool cvShowManyImages(char const* title, vector<Mat> images);


void imageProcessing(const sensor_msgs::ImageConstPtr& msg){
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
    images.push_back(cv_ptr->image);
	cvShowManyImages("Loop closure!", images);
	images.clear();
}


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

void loopProcessing(const std_msgs::Int32MultiArray::ConstPtr& imgs){
	Mat img_current;
	for(std::vector<int>::const_iterator it = imgs->data.begin()+1; it != imgs->data.end(); ++it)
	{
     string img_path = (*(imageNames.begin()+*(it))).string();

        img_current = imread(img_path);
        images.push_back(img_current);
	}
	ROS_INFO("A new loop closure has been detected!");
	//cvShowManyImages("Loop closure!", images);
	//images.clear();
}

bool cvShowManyImages(char const* title, vector<Mat> images) {
	Mat img;
	Mat DispImage;
	int subImageSize = 200;
	int i,m,n;
	int imgCols, imgRows;
	int w = 3, h = 2;
	float scale;
	int max;
	if(images.size() <= 0) {
		ROS_ERROR("Number of images too small!");
		return false;
	}
	else if(images.size() > 6) {
		ROS_ERROR("Number of images too large!");
		return false;
	}
	DispImage = cvCreateImage(cvSize(100 + subImageSize*w, 60 + subImageSize*h),8,3);
	for (i = 0, m = 20, n = 20; i < images.size(); i++, m += (20 + subImageSize)) {
		img = images[i];
		if(img.empty()) {
			ROS_ERROR("Invalid image!");
			return false;
		}
		imgCols = img.cols;
		imgRows = img.rows;
		max = (imgCols > imgRows)? imgCols: imgRows;
		scale = (float) ( (float) max / subImageSize );
		if( i % w == 0 && m!= 20) {
			m = 20;
			n+= 20 + subImageSize;
		}
		Rect roi(m, n, (int)(imgCols/scale), (int)(imgRows/scale));
		Mat image_roi = DispImage(roi);
		resize(img, image_roi, image_roi.size(), 0, 0,0);
	}
	namedWindow(title, 1);
	imshow(title,DispImage);
	if(images.size()==1)
		waitKey(10);
	else
		waitKey(0);

	return true;
}


int main(int argc, char **argv)
{
	if(argc != 3){
		ROS_ERROR("Improper use of function parameters!");
		ROS_ERROR("Usage: ./imageViewer directory fileExtension");
		return -1;
	}
	int result = get_all(argv[1], argv[2], imageNames);
	if( result == -1){
		ROS_ERROR("Non existing directory!");
		return -1;
	}
	else if (result == 0){
		ROS_ERROR("Empty directory!");
		return -1;
	}

	sort(imageNames.begin(),imageNames.end());

	std::ofstream myfile;
	myfile.open("outData/param/imageViewer.txt");
	myfile << argv[0]<<" "<< argv[1] << " " << argv[2];
	myfile.close();

	ros::init(argc, argv, "imageViewer");
	ros::NodeHandle n;
	ros::Subscriber img_sub = n.subscribe<sensor_msgs::Image>("/images", 100, imageProcessing);
	ros::Subscriber loop_sub = n.subscribe<std_msgs::Int32MultiArray>("/loopClosures",100, loopProcessing);
	ros::spin();
  	return 0;
}


