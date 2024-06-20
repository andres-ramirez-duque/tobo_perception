#ifndef _PERCEPTIONSTREAM_PERCEPTIONSTREAM_H
#define __PERCEPTIONSTREAM_PERCEPTIONSTREAM_H

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <locale.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <gmodule.h>
#include "gstnvdsmeta.h"
#include "gst-nvmessage.h"
#include "nvds_version.h"
#include "nvdsmeta.h"
#include "nvdsinfer.h"
#include "nvdsinfer_custom_impl.h"
#include "gstnvdsinfer.h"
#include "cuda_runtime_api.h"
#include "ds_facialmark_meta.h"
#include "ds_gaze_meta.h"
#include "cv/core/Tensor.h"
#include "nvbufsurface.h"
#include <map>
#include <fstream>
#include <iostream>
#include <sstream>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "LandmarkCoreIncludes.h"
#include "FaceAnalyser.h"
#include "GazeEstimation.h"
#include "Visualizer.h"
#include "VisualizationUtils.h"

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <std_msgs/String.h>
#include <std_msgs/Int64.h>
#include <std_srvs/Trigger.h>
#include "hri_msgs/FacialLandmarks.h"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>


#include <sensor_msgs/CameraInfo.h>


#include <stdexcept>

namespace tobo_perception {

  class PerceptionStream {
  public:
    PerceptionStream(ros::NodeHandle nodeh);
    ~PerceptionStream();

    bool configure();
    bool init_stream();
    void publish_stream();
    void cleanup_stream();
    void run();
    void stopRecording();
    void startRecording();
    bool toggle_recording(std_srvs::Trigger::Request& request, std_srvs::Trigger::Response& response);
    

    
    
    struct perf_measure{
        GstClockTime pre_time;
        GstClockTime total_time;
        guint count;
    };
    
    struct DsSourceBin{
      GstElement *source_bin;
      GstElement *uri_decode_bin;
      GstElement *vidconv;
      GstElement *nvvidconv;
      GstElement *capsfilt;
      gint index;
    };
    
    DsSourceBin source_struct;
    perf_measure perf_measure;
    guint src_cnt;
    guint bus_watch_id;
    std::string video_name;
    std::string configs_path;
    std::string recordings_path;
    std::string config_primary;
    std::string config_secondary;
    std::string config_gaze;
    std::string config_hr;
    std::string config_emotion;
    std::string config_tracker;
    std::string gaze_lib;
    std::string hr_lib;
    std::string emotion_lib;
    std::string tracker_lib;
    
    int ros_hz;
    std::string face_frame_id;
    std::string camera_frame_id;
    std::string world_frame_id;
    
    double time_offset_;
    
    ros::NodeHandle nh;
    image_transport::ImageTransport image_transport;
    image_transport::Publisher camera_pub;
    
    
    sensor_msgs::ImagePtr msg;
    ros::Publisher gaze_pub;
    ros::Publisher emotion_pub;
    ros::Publisher landmarks_pub;
    ros::Publisher hrate_pub;
    tf2_ros::TransformBroadcaster tf_br;
    ros::ServiceServer service;
    
    int zed_resolution;
    guint stream_width;
    guint stream_height;
    bool os_display;
    bool src_zed_gst;
    guint frame_width;
    guint frame_height;
    
    cv::Mat cvmat;
    std::vector<geometry_msgs::TransformStamped> face_transform;
    
    std_msgs::String emotion;
    std_msgs::Int64 h_rate;
    hri_msgs::FacialLandmarks FacialLandmarks;
    hri_msgs::NormalizedPointOfInterest2D p;
    
    LandmarkDetector::CLNF face_model;
    std::string file_uri;
    bool src_file;
    
    std::vector<int> mappings;
    double fx;
    double fy;
    double cx;
    double cy;
    int n_part_points;
    std::vector<cv::Vec3d> rotation_hypotheses;
    cv::Mat_<float> old_landmarks;
    cv::Vec3f rotation_init;
    cv::Vec2f translation;
    float scaling;
    
    cv::Vec6d head_pose;
    
    // Gstreamer structures
    GstElement *pipeline_;
    GMainLoop *loop = NULL;
    GstElement *tee, *rec_videoconvert, *rec_capsfilt, *rec_encoder, *rec_muxer, *rec_filesink, *queue_record, *h264parse, *omx_capsfilt;
    GstPad *teepad;
    const char *file_path;
    const char *vid_name;
    gint counter = 0;
    gboolean recording = FALSE;
      
  private:
    
    // Camera publisher configuration
    std::string frame_id_;
    int width_, height_;
    std::string image_encoding_;
    std::string camera_name_;
    std::string camera_info_url_;

    // ROS Inteface
    // Calibration between ros::Time and gst timestamps
    
    // Case of a jpeg only publisher
    ros::Publisher jpeg_pub_;
    ros::Publisher cinfo_pub_;
  };

}

#endif // ifndef __PERCEPTIONSTREAM_PERCEPTIONSTREAM_H
