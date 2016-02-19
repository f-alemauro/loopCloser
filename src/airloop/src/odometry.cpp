#include "ros/ros.h"
#include "std_msgs/Time.h"
#include "sensor_msgs/Imu.h"
#include <geometry_msgs/Vector3.h>
#include <fstream>

using namespace std;
using namespace ros;

ros::Time ts;

void newTs(const std_msgs::Time time){
	cout<<"newTS"<<endl;
	ts = time.data;
}

int main(int argc, char **argv)
{

	srand (time(NULL));
	int loopRate=10;
	sensor_msgs::Imu data;

	ofstream myfile;
	myfile.open ("outData/param/odometry.txt",std::ofstream::app);
	for (int i = 0; i< argc;i++)
		myfile << argv[i]<<" ";
	myfile<<"\n";
	myfile.close();

	ros::init(argc, argv, "dummyOdometry");
	NodeHandle n;
	Publisher odom_pub = n.advertise<sensor_msgs::Imu>("/odom", 100);
	Subscriber ts_sub = n.subscribe<std_msgs::Time>("/timeStamp", 1, newTs);
	if(odom_pub){
		Rate loop_rate(loopRate);
		while(odom_pub.getNumSubscribers()<1 && ok())
			ROS_INFO("Waiting for subscribers...");

		while(odom_pub.getNumSubscribers()>0 && ok()){

			int dummyOdom = rand() % 125 + 1;


			geometry_msgs::Vector3 newOmega;
			newOmega.x = dummyOdom;
			newOmega.y = dummyOdom;
			newOmega.z = dummyOdom;

			data.linear_acceleration = newOmega;
			data.header.stamp = ts;
			ROS_INFO("ODOMETRY DATA: %d-%d-%d",(int)data.linear_acceleration.x,(int)data.linear_acceleration.y,(int)data.linear_acceleration.z);

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
