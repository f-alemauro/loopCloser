#include "ros/ros.h"
#include <std_msgs/Int8.h>
using namespace std;
using namespace ros;

int main(int argc, char **argv)
{
	srand (time(NULL));
	int loopRate=10;
	std_msgs::Int8 data;

	ros::init(argc, argv, "dummyOdometry");
	NodeHandle n;
	Publisher odom_pub = n.advertise<std_msgs::Int8>("/odom", 100);
	if(odom_pub){
		Rate loop_rate(loopRate);
		while(odom_pub.getNumSubscribers()<1 && ok())
			ROS_INFO("Waiting for subscribers...");
		while(odom_pub.getNumSubscribers()>0 && ok()){
			int dummyOdom = rand() % 125 + 1;
			data.data = dummyOdom;
			ROS_INFO("ODOMETRY DATA: %d", data.data);
			odom_pub.publish(data);
			spinOnce();
			loop_rate.sleep();
		}
		if (odom_pub.getNumSubscribers()==0){
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
