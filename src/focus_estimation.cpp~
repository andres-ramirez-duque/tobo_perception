#include <string>
#include <vector>
#include <sstream>

#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/Range.h>
#include <std_msgs/ColorRGBA.h>
#include <std_msgs/String.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <numeric>

#include <tobo_planner/SensorValue.h>

using namespace std;
static const double FOV = 45. / 180. * M_PI; // radians
static const float RANGE = 4.5; //m
geometry_msgs::TransformStamped prev_pose;
vector<float> win_angular;
vector<string> on_focus_frames;
std::map<std::string, std::size_t> results;
int win_number=0;
int win_eng_number=0;

bool readsensor(tobo_planner::SensorValue::Request  &req,
         tobo_planner::SensorValue::Response &res)
{
  res.sensor_value = false;
  float i = 0.0, j = 0.0;
  std::for_each(on_focus_frames.begin(), on_focus_frames.end(), [&](std::string const& s)
  {
    ROS_INFO("value: %s", s.c_str());
    if (s.compare("True") == 0)
    {
      i = i + 1.0f;
    }
    j = j + 1.0f;
  });
  res.sensor_value = (i/j > 0.5);
  ROS_INFO("i: %f, j: %f", i, j);
  if (res.sensor_value)
  {  
    ROS_INFO("sensor_value: True");
  }
  else
  {
    ROS_INFO("sensor_value: False");
  }
  //int attention_cam = int(results.find("zed2_camera_center")->second);
  std::string sensor_msg("Sensing engagement");
  res.message = sensor_msg;
  ROS_INFO("request: %s", req.request_type.c_str());
  ROS_INFO("sending back response: [%d]", res.sensor_value);
  return true;
}

visualization_msgs::Marker makeMarker(const string& frame) {
    static std_msgs::ColorRGBA Purple;
    Purple.r = 0.5; Purple.g = 0.; Purple.b = 0.5; Purple.a = 1.;
    
    visualization_msgs::Marker marker;
    marker.header.frame_id = frame;
    marker.header.stamp = ros::Time::now();
    marker.ns = "focus_of_attention";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = 0;
    marker.pose.position.y = 0;
    marker.pose.position.z = 0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.10;
    marker.scale.y = 0.10;
    marker.scale.z = 0.10;
    marker.color = Purple;
    marker.lifetime = ros::Duration(2);
    return marker;
}

bool isInFieldOfView(const tf2_ros::Buffer& listener, const string& target_frame, const string& observer_frame) {

    geometry_msgs::TransformStamped transform;

    try{
      transform = listener.lookupTransform(observer_frame, target_frame, ros::Time::now(), ros::Duration(3.0));
    }
    catch(tf2::TransformException &ex){
      ROS_WARN("%s",ex.what());
    }
    if (transform.transform.translation.x < 0) return false;
   
    double distance_to_main_axis = sqrt(transform.transform.translation.y * transform.transform.translation.y + transform.transform.translation.z * transform.transform.translation.z);
    double fov_radius_at_x = tan(FOV/2) * transform.transform.translation.x;
    
    if (distance_to_main_axis < fov_radius_at_x) return true;
    return false;
}
void angular_v_estimation(const geometry_msgs::TransformStamped& current_pose) {
    
    if (current_pose.transform.translation.x != 0.0 & current_pose.transform.translation.x != 0.0 )
    {
      float fs = 1.0;
      geometry_msgs::Quaternion prev_q_msgs; 
      tf2::Quaternion curr_q, delta_q, prev_q;
      prev_q_msgs = prev_pose.transform.rotation;
      prev_q_msgs.w = -prev_q_msgs.w;
      tf2::convert(prev_q_msgs, prev_q);
      tf2::convert(current_pose.transform.rotation, curr_q);
      delta_q = curr_q*prev_q;
      delta_q.normalize();
      tf2::Vector3 v_angular = tf2::Vector3(delta_q.getX(), delta_q.getY(), delta_q.getZ());
      tf2Scalar v_angular_len = v_angular.length();
      tf2Scalar delta_q_angle = tf2Atan2(v_angular_len, delta_q.getW());
      tf2::Vector3 angular = v_angular * delta_q_angle * fs;
    
      win_angular.push_back(float(angular.length()));
      ROS_INFO("Relative Angular Velocity (Omega) = %f; V_len = %f", float(angular.length()), float(v_angular_len));
      prev_pose = current_pose;
    }
}

void win_processing(const ros::TimerEvent&)
{
    if (!win_angular.empty()){
      float average = accumulate( win_angular.begin(), win_angular.end(), 0.0)/win_angular.size();  
      ROS_INFO("Mean Angular Velocity (1s) = %f; Size= %d \n", average, int(win_angular.size()));
      win_angular.erase(win_angular.begin(),win_angular.end()-1);
    }
    
    results.clear();
    if (!on_focus_frames.empty()){
      std::for_each(on_focus_frames.begin(), on_focus_frames.end(), [&](std::string const& s)
      {
        ++results[s];
      });
      for( auto p : results ) {
        if( (p.first.compare("nao_frame_id") != 0) && (int(p.second)>0))
          win_eng_number++;
        ROS_INFO("frame = %s; Qty= %d \n", p.first.c_str(), int(p.second));
      }
      on_focus_frames.clear();
    }
    win_number++;
}

int main( int argc, char** argv )
{
    ros::init(argc, argv, "estimate_focus");
    ros::NodeHandle n("~");
    ros::Rate r(1); //5Hz
    //ros::Timer timer1 = n.createTimer(ros::Duration(2), win_processing); //2 seconds 
    ros::Publisher marker_pub = n.advertise<visualization_msgs::Marker>("estimate_focus", 1);
    ros::Publisher fov_pub = n.advertise<sensor_msgs::Range>("face_field_of_view", 1);
    ros::Publisher frames_in_fov_pub = n.advertise<std_msgs::Header>("actual_focus_of_attention", 1);
    
    ros::ServiceServer sensor_service = n.advertiseService("/get_sensor_value", readsensor);
    
    std::string focus_frame_id;
    n.param("/focus_frame_id", focus_frame_id, std::string("focus_frame"));
    
    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener tfListener(tfBuffer);
    geometry_msgs::TransformStamped focus_frame_tr;

    vector<string> monitored_frames = {"nao_frame_id"};//, "zed2_camera_center"

    std_msgs::Header fov_header;
    sensor_msgs::Range fov;
    
    fov.radiation_type = sensor_msgs::Range::INFRARED;
    fov.field_of_view = FOV;
    fov.min_range = 0;
    fov.max_range = 10;
    fov.range = RANGE;
    bool face_visible=false;
    ROS_INFO("Waiting until a face becomes visible...");
    /*
    bool isready;
    while (!ros::param::get("/user_is_ready", isready))
    {
      ROS_INFO_ONCE("Waiting for nodes initialization");
    }*/
    while (!face_visible) {
      try{
        focus_frame_tr = tfBuffer.lookupTransform("world", focus_frame_id, ros::Time::now(), ros::Duration(3.0));
        face_visible=true;
      }
      catch(tf2::TransformException &ex){
        ROS_WARN("%s",ex.what());
      }
      r.sleep();
    }    
    ros::Duration(30).sleep(); // XXX Added to compensate for long start up..
    ROS_INFO("Face detected! We can start estimating the focus of attention...");
    int time_slots = 0, looking_at_target = 0;
    while (ros::ok())
    {
      try{
        focus_frame_tr = tfBuffer.lookupTransform("world", focus_frame_id, ros::Time::now(), ros::Duration(1.0)); // XXX CHANGED from 3.0 - should not delay data collection here?
        angular_v_estimation (focus_frame_tr);
      }
      catch(tf2::TransformException &ex){
        ROS_WARN("%s",ex.what());
      }
      
      for(auto frame : monitored_frames) {
        if(isInFieldOfView(tfBuffer, frame, focus_frame_id)) {
          on_focus_frames.push_back("True");
          fov_header.frame_id=frame;
          fov_header.stamp=ros::Time::now();
          marker_pub.publish(makeMarker(frame));
          frames_in_fov_pub.publish(fov_header);
          ROS_INFO("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Looking at target!");
          ++looking_at_target;
        }else{
          on_focus_frames.push_back("False");
          ROS_INFO("################################# NOT looking at target!");
        } 
        ++time_slots; 
      }
      if(on_focus_frames.size() > 60)
      {
        on_focus_frames.erase(on_focus_frames.begin());
      }
      fov.range = RANGE;
      fov.header.stamp = ros::Time::now();
      fov.header.frame_id = focus_frame_id;
      fov_pub.publish(fov); 
      ros::spinOnce();
      r.sleep();  
    }
    ROS_INFO("Total number of time slots: %d, and total number engaged: %d \n", time_slots, looking_at_target);
    ROS_INFO("Totally %d win analyzed \n", win_number);
    ROS_INFO("Totally %d engagement detected \n", win_eng_number);
    ROS_INFO("Percentage %f engagement detected \n", win_eng_number/win_number);
}
