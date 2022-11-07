#include <ros/ros.h>
#include <tobo_perception/perceptionstream.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "perceptionstream_publisher");
  ros::NodeHandle nh;

  tobo_perception::PerceptionStream perceptionstream_driver(nh);
  perceptionstream_driver.run();

  return 0;
}
