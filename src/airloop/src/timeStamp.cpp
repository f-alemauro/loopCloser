#include "ros/ros.h"

#include "std_msgs/Time.h"
#include <fstream>

using namespace std;
using namespace ros;

int main(int argc, char **argv)
{
	std_msgs::Time t;
	ros::Time actualTime;
	actualTime.useSystemTime();
	if(argc < 2){
		ROS_ERROR("Missing loop rate! Correct usage is ./timeStamp loopRate");
		return -1;
	}
	int loopRate = atoi(argv[1]);


	ofstream myfile;
	myfile.open ("outData/param/timeStamp.txt",std::ofstream::app);
	for (int i = 0; i< argc;i++)
		myfile << argv[i]<<" ";
	myfile<<"\n";
	myfile.close();

	ros::init(argc, argv, "timeStamp");
	NodeHandle n;
	Publisher ts_pub = n.advertise<std_msgs::Time>("/timeStamp", 1);
	if(ts_pub){
		Rate loop_rate(loopRate);
		ROS_INFO("Starting to publish timestamp...");

		while(ros::ok()){
			t.data = actualTime.now();
			cout<<"new data gone"<<endl;
			ts_pub.publish(t);
			loop_rate.sleep();
		}
		ROS_ERROR("Error in ROS! Qutting!");
		return -1;
	}
	else{
		ROS_ERROR("Error in creating publisher!");
		return -1;
	}
	return 0;
}
