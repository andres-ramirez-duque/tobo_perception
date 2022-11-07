#!/usr/bin/env python  
import rospy
import tf2_ros
import geometry_msgs.msg


if __name__ == '__main__':
    rospy.init_node('calib_node')

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    calib_file_name = rospy.get_param('calib_file_name','scenario_calib.launch')
    charuco_zed_frame_id = rospy.get_param('charuco_zed_frame_id','charuco')
    charuco_nao_frame_id = rospy.get_param('charuco_nao_frame_id','charuco')
    camera_frame_id = rospy.get_param('camera_frame_id','zed2_camera_center')
    tobo_frame_id = rospy.get_param('tobo_frame_id','nao_frame_id')
    calib_time = rospy.get_param('calib_time', 5.0)
    write_file=False
    rate = rospy.Rate(1.0)
    i=0
    
    s = "<launch>\n"
    
    s += "  <node name=\"focus_frame\" pkg=\"tf2_ros\" type=\"static_transform_publisher\" args=\"0 0 0 0 0 0"
    s += "    world child_gaze\"/>\n\n"
    
    s += "  <node name=\"temporal_frame\" pkg=\"tf2_ros\" type=\"static_transform_publisher\" args=\"0 0 0 0 -1.570758 0"
    s += "    child_gaze focus_frame\"/>\n\n"
  
    while not rospy.is_shutdown():
        try:
            zed2_trans = tfBuffer.lookup_transform(charuco_zed_frame_id, camera_frame_id, rospy.Time())
            
            s += "  <node name=\"zed2\" pkg=\"tf2_ros\" type=\"static_transform_publisher\" args=\"" + str(zed2_trans.transform.translation.x) + "\n"
            s += "    " + str(zed2_trans.transform.translation.y) + "\n"
            s += "    " + str(-zed2_trans.transform.translation.z) + "\n"
            s += "    " + str(zed2_trans.transform.rotation.x) + "\n"
            s += "    " + str(zed2_trans.transform.rotation.y) + "\n"
            s += "    " + str(zed2_trans.transform.rotation.z) + "\n"
            s += "    " + str(zed2_trans.transform.rotation.w) + "\n"
            s += "    world " + camera_frame_id + "\"/>\n\n"
            
            write_file=True
            rospy.loginfo("transform Charuco to Camera was found");
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.loginfo("Could NOT transform Charuco to camera: ");
        
        try:
            nao_trans = tfBuffer.lookup_transform(charuco_nao_frame_id, tobo_frame_id, rospy.Time())
            
            s += "  <node name=\"zed2\" pkg=\"tf2_ros\" type=\"static_transform_publisher\" args=\"" + nao_trans.transform.translation.x + "\n"
            s += "    " + str(nao_trans.transform.translation.y) + "\n"
            s += "    " + str(-nao_trans.transform.translation.z) + "\n"
            s += "    " + str(nao_trans.transform.rotation.x) + "\n"
            s += "    " + str(nao_trans.transform.rotation.y) + "\n"
            s += "    " + str(nao_trans.transform.rotation.z) + "\n"
            s += "    " + str(nao_trans.transform.rotation.w) + "\n"
            s += "    world" + tobo_frame_id + "\"/>\n\n"
            
            write_file=True
            rospy.loginfo("transform Charuco to NAO was found");
                     
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.loginfo("Could NOT transform Charuco to NAO");
        
        if write_file == True and i > calib_time:
          with open(calib_file_name, 'w') as file:
            s += "</launch>"
            file.write(s)         
            file.close()
            break
        rate.sleep()
        i += 1
