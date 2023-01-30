
#include <stdlib.h>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#include <ros/ros.h>

#include <image_transport/image_transport.h>



#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/SetCameraInfo.h>
#include <sensor_msgs/image_encodings.h>

#include <tobo_perception/perceptionstream.h>

using namespace std;
using std::string;

#define MAX_DISPLAY_LEN 64

#define MEASURE_ENABLE 1

#define PGIE_CLASS_ID_FACE 0

#define PGIE_DETECTED_CLASS_NUM 4

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1280
#define MUXER_OUTPUT_HEIGHT 720

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 400000

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing cuda buffers. */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"
#define CONFIG_GPU_ID "gpu-id"

#define SGIE_NET_WIDTH 80
#define SGIE_NET_HEIGHT 80

guint frame_number = 0;
guint total_face_num = 0;

#define PRIMARY_DETECTOR_UID 1
#define SECOND_DETECTOR_UID 2

std::unique_ptr<cvcore::faciallandmarks::FacialLandmarksPostProcessor> facemarkpost;
namespace tobo_perception {
  
  static geometry_msgs::Quaternion toQuaternion(double pitch, double roll, double yaw)
  {
    double t0 = std::cos(yaw * 0.5f);
    double t1 = std::sin(yaw * 0.5f);
    double t2 = std::cos(roll * 0.5f);
    double t3 = std::sin(roll * 0.5f);
    double t4 = std::cos(pitch * 0.5f);
    double t5 = std::sin(pitch * 0.5f);

    geometry_msgs::Quaternion q;
    q.w = t0 * t2 * t4 + t1 * t3 * t5;
    q.x = t0 * t3 * t4 - t1 * t2 * t5;
    q.y = t0 * t2 * t5 + t1 * t3 * t4;
    q.z = t1 * t2 * t4 - t0 * t3 * t5;
    return q;
  }

  static geometry_msgs::Quaternion operator *(const geometry_msgs::Quaternion &a, const geometry_msgs::Quaternion &b)
  {
    geometry_msgs::Quaternion q;
    
    q.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;  // 1
    q.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;  // i
    q.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;  // j
    q.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;  // k
    return q;
  }
  
  static cv::Matx33f Euler2RotationMatrix(const cv::Vec3f& eulerAngles)
  {
		cv::Matx33f rotation_matrix;

		float s1 = sin(eulerAngles[0]);
		float s2 = sin(eulerAngles[1]);
		float s3 = sin(eulerAngles[2]);

		float c1 = cos(eulerAngles[0]);
		float c2 = cos(eulerAngles[1]);
		float c3 = cos(eulerAngles[2]);

		rotation_matrix(0, 0) = c2 * c3;
		rotation_matrix(0, 1) = -c2 *s3;
		rotation_matrix(0, 2) = s2;
		rotation_matrix(1, 0) = c1 * s3 + c3 * s1 * s2;
		rotation_matrix(1, 1) = c1 * c3 - s1 * s2 * s3;
		rotation_matrix(1, 2) = -c2 * s1;
		rotation_matrix(2, 0) = s1 * s3 - c1 * c3 * s2;
		rotation_matrix(2, 1) = c3 * s1 + c1 * s2 * s3;
		rotation_matrix(2, 2) = c1 * c2;

		return rotation_matrix;
  }
  /* Calculate performance data */
  static GstPadProbeReturn
  osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
        gpointer u_data)
    {
    GstBuffer *buf = (GstBuffer *) info->data;
    NvDsObjectMeta *obj_meta = NULL;
    guint face_count = 0;
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    
    int frame_width, frame_height;
    cv::Mat frame;  
    NvBufSurface *in_surf = NULL;
    GstMapInfo in_map_info;
  
    PerceptionStream *perception = (PerceptionStream *)(u_data);
    
    memset (&in_map_info, 0, sizeof (in_map_info));

    /* Map the buffer contents and get the pointer to NvBufSurface. */
    if (!gst_buffer_map (buf, &in_map_info, GST_MAP_READ)) {
        g_printerr ("Failed to map GstBuffer\n");
        gst_buffer_unmap (buf, &in_map_info);
        return GST_PAD_PROBE_PASS;
    }
    in_surf = (NvBufSurface *) in_map_info.data;
    NvBufSurfaceMap(in_surf, -1, -1, NVBUF_MAP_READ_WRITE);
    NvBufSurfaceSyncForCpu(in_surf, -1, -1);
      
    GstClockTime now;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    if (frame_number>10){
    now = g_get_monotonic_time();
    if (perception->perf_measure.pre_time == GST_CLOCK_TIME_NONE) {
        perception->perf_measure.pre_time = now;
        perception->perf_measure.total_time = GST_CLOCK_TIME_NONE;
    } else {
        if (perception->perf_measure.total_time == GST_CLOCK_TIME_NONE) {
        perception->perf_measure.total_time = (now - perception->perf_measure.pre_time);
        } else {
        perception->perf_measure.total_time += (now - perception->perf_measure.pre_time);
        }
        perception->perf_measure.pre_time = now;
        perception->perf_measure.count++;
    }
    }
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
        l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        
        frame_width = in_surf->surfaceList[frame_meta->batch_id].width;
        frame_height = in_surf->surfaceList[frame_meta->batch_id].height;
        void *frame_data = in_surf->surfaceList[frame_meta->batch_id].mappedAddr.addr[0];
        size_t frame_step = in_surf->surfaceList[frame_meta->batch_id].pitch;
        
        //frame = cv::Mat(frame_height, frame_width, CV_8UC4, frame_data, frame_step);
                   
        if (!frame_meta)
            continue;

        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
            l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta *) (l_obj->data);

            if (!obj_meta)
                continue;
      /* Check that the object has been detected by the primary detector
      * and that the class id is that of vehicles/persons. */
            if (obj_meta->unique_component_id == PRIMARY_DETECTOR_UID) {
                if (obj_meta->class_id == PGIE_CLASS_ID_FACE)
                    face_count++;
            }
        }
    }
    
    /*try
    {
        perception->msg = cv_bridge::CvImage(std_msgs::Header(), "rgba8", frame).toImageMsg();
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return GST_PAD_PROBE_PASS;
    } */ 
    
    NvBufSurfaceUnMap(in_surf, -1, -1);
    gst_buffer_unmap(GST_BUFFER (info->data), &in_map_info);
    //g_print ("Frame Number = %d Face Count = %d\n", frame_number, face_count);
    frame_number++;
    total_face_num += face_count;
    return GST_PAD_PROBE_OK;
  }

/*Generate bodypose2d display meta right after inference */
  static GstPadProbeReturn
    tile_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
        gpointer u_data)
    {
    GstBuffer *buf = (GstBuffer *) info->data;
    NvDsObjectMeta *obj_meta = NULL;
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    int part_index = 0;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
    PerceptionStream *perception = (PerceptionStream *)(u_data);
    
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
        l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);    
        NvDsDisplayMeta *disp_meta = NULL;
        
        if (!frame_meta)
        continue;

        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
         l_obj = l_obj->next) {
        obj_meta = (NvDsObjectMeta *) (l_obj->data);

        if (!obj_meta)
            continue;

        bool facebboxdraw = false;
                       
        for (NvDsMetaList * l_user = obj_meta->obj_user_meta_list;
            l_user != NULL; l_user = l_user->next) {
            NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
            if(user_meta->base_meta.meta_type ==
                (NvDsMetaType)NVDS_USER_RIVA_META_FACEMARK) {
              NvDsFacePointsMetaData *facepoints_meta =
                (NvDsFacePointsMetaData *)user_meta->user_meta_data;
              facepoints_meta->facemark_num = perception->n_part_points;
          
              /*Get the face marks and mark with dots*/
              if (!facepoints_meta)
                continue;
              for (part_index = 0;part_index < facepoints_meta->facemark_num;
                part_index++) {
                if (!disp_meta) {
                  disp_meta = nvds_acquire_display_meta_from_pool(batch_meta);
                  disp_meta->num_circles = 0;
                  disp_meta->num_rects = 0;
                  disp_meta->num_labels = 0;
              
                } else {
                  if (disp_meta->num_circles==MAX_ELEMENTS_IN_DISPLAY_META) {
                
                    nvds_add_display_meta_to_frame (frame_meta, disp_meta);
                    disp_meta = nvds_acquire_display_meta_from_pool(batch_meta);
                    disp_meta->num_circles = 0;
                    disp_meta->num_labels = 0;
                  }
                }
                if(!facebboxdraw) {
                  disp_meta->rect_params[disp_meta->num_rects].left =
                    facepoints_meta->right_eye_rect.left +
                    obj_meta->rect_params.left;
                  disp_meta->rect_params[disp_meta->num_rects].top =
                    facepoints_meta->right_eye_rect.top +
                    obj_meta->rect_params.top;
                  disp_meta->rect_params[disp_meta->num_rects].width =
                    facepoints_meta->right_eye_rect.right -
                    facepoints_meta->right_eye_rect.left;
                  disp_meta->rect_params[disp_meta->num_rects].height =
                    facepoints_meta->right_eye_rect.bottom -
                    facepoints_meta->right_eye_rect.top;
                  disp_meta->rect_params[disp_meta->num_rects].border_width = 2;
                  disp_meta->rect_params[disp_meta->num_rects].border_color.red = 1.0;
                  disp_meta->rect_params[disp_meta->num_rects].border_color.green = 1.0;
                  disp_meta->rect_params[disp_meta->num_rects].border_color.blue = 0.0;
                  disp_meta->rect_params[disp_meta->num_rects].border_color.alpha = 0.5;
                  disp_meta->rect_params[disp_meta->num_rects+1].left =
                    facepoints_meta->left_eye_rect.left + obj_meta->rect_params.left;
                  disp_meta->rect_params[disp_meta->num_rects+1].top =
                    facepoints_meta->left_eye_rect.top + obj_meta->rect_params.top;
                  disp_meta->rect_params[disp_meta->num_rects+1].width =
                    facepoints_meta->left_eye_rect.right -
                    facepoints_meta->left_eye_rect.left;
                  disp_meta->rect_params[disp_meta->num_rects+1].height =
                    facepoints_meta->left_eye_rect.bottom -
                    facepoints_meta->left_eye_rect.top;
                  disp_meta->rect_params[disp_meta->num_rects+1].border_width = 2;
                  disp_meta->rect_params[disp_meta->num_rects+1].border_color.red = 1.0;
                  disp_meta->rect_params[disp_meta->num_rects+1].border_color.green = 1.0;
                  disp_meta->rect_params[disp_meta->num_rects+1].border_color.blue = 0.0;
                  disp_meta->rect_params[disp_meta->num_rects+1].border_color.alpha = 0.5;
                  
                  disp_meta->num_rects+=2;
                  facebboxdraw = true;
                }
                  
                disp_meta->circle_params[disp_meta->num_circles].xc =
                  facepoints_meta->mark[part_index].x + obj_meta->rect_params.left;
                disp_meta->circle_params[disp_meta->num_circles].yc =
                  facepoints_meta->mark[part_index].y + obj_meta->rect_params.top;
                disp_meta->circle_params[disp_meta->num_circles].radius = 1;
                disp_meta->circle_params[disp_meta->num_circles].circle_color.red = 0.0;
                disp_meta->circle_params[disp_meta->num_circles].circle_color.green = 1.0;
                disp_meta->circle_params[disp_meta->num_circles].circle_color.blue = 0.0;
                disp_meta->circle_params[disp_meta->num_circles].circle_color.alpha = 0.5;
                disp_meta->num_circles++;
              }
          
              cv::Rect_<float> bounding_box (obj_meta->rect_params.left, 
                obj_meta->rect_params.top, obj_meta->rect_params.width, obj_meta->rect_params.height);
	            
	            perception->old_landmarks.setTo(0);
	            
              for (int mapping_ind = 0; mapping_ind < perception->n_part_points; mapping_ind++) {
                perception->old_landmarks.at<float>(mapping_ind) =
                 facepoints_meta->mark[mapping_ind].x + obj_meta->rect_params.left;
		 	          perception->old_landmarks.at<float>(mapping_ind + perception->n_part_points) =
		 	           facepoints_meta->mark[mapping_ind].y + obj_meta->rect_params.top;
              }
              float currErru=1000;
              size_t best_hypothesis=0;
              for (size_t hypothesis = 0; hypothesis < perception->rotation_hypotheses.size(); ++hypothesis){
	              perception->face_model.Reset();  
                perception->face_model.params_local.setTo(0);
                perception->face_model.pdm.CalcParams(perception->face_model.params_global, 
                  bounding_box,   perception->face_model.params_local);
                perception->face_model.pdm.CalcParams(perception->face_model.params_global, 
                  perception->face_model.params_local, perception->old_landmarks, perception->rotation_hypotheses[hypothesis]);		
		            perception->face_model.pdm.CalcShape2D(perception->face_model.detected_landmarks, 
		              perception->face_model.params_local, perception->face_model.params_global);
		    
		            perception->scaling = perception->face_model.params_global[0];
		            perception->rotation_init[0] = perception->face_model.params_global[1];
		            perception->rotation_init[1] = perception->face_model.params_global[2];
		            perception->rotation_init[2] = perception->face_model.params_global[3];

		            perception->translation[0] = perception->face_model.params_global[4];
		            perception->translation[1] = perception->face_model.params_global[5];
		            cv::Matx33f R  = Euler2RotationMatrix(perception->rotation_init);
		            cv::Matx23f R_2D(R(0,0), R(0,1), R(0,2), R(1,0), R(1,1), R(1,2));
		            cv::Mat_<float> shape_3D;
		    
		            perception->face_model.pdm.CalcShape3D(shape_3D,perception->face_model.params_local);
		            shape_3D = shape_3D.reshape(1, 3);
                cv::Mat_<float> curr_shape_2D = perception->scaling * shape_3D.t() * cv::Mat(R_2D).t();
                curr_shape_2D.col(0) = curr_shape_2D.col(0) + perception->translation(0);
		            curr_shape_2D.col(1) = curr_shape_2D.col(1) + perception->translation(1);

		            curr_shape_2D = cv::Mat(curr_shape_2D.t()).reshape(1, perception->n_part_points * 2);  
                float error = cv::norm(curr_shape_2D - perception->old_landmarks);
                if(error < currErru){
                  currErru=error;
                  best_hypothesis=hypothesis;
                }
	            }
	            perception->face_model.Reset();  
              perception->face_model.params_local.setTo(0);
              perception->face_model.pdm.CalcParams(perception->face_model.params_global, bounding_box, perception->face_model.params_local);
              perception->face_model.pdm.CalcParams(perception->face_model.params_global, 
                perception->face_model.params_local, perception->old_landmarks, perception->rotation_hypotheses[best_hypothesis]);		
		          perception->face_model.pdm.CalcShape2D(perception->face_model.detected_landmarks, perception->face_model.params_local, perception->face_model.params_global);
              perception->head_pose = LandmarkDetector::GetPose(perception->face_model, perception->fx, perception->fy, perception->frame_width/2, perception->frame_height/2);
		                      
            } else if(user_meta->base_meta.meta_type ==
              (NvDsMetaType)NVDS_USER_RIVA_META_GAZE) {
              NvDsGazeMetaData * gazemeta =
                  (NvDsGazeMetaData *)user_meta->user_meta_data;
              g_print("Gaze:");
              for (int i=0; i<cvcore::gazenet::GazeNet::OUTPUT_SIZE; i++){
                g_print(" %f", gazemeta->gaze_params[i]);
              }
            }
          }         
        }
        if (disp_meta && disp_meta->num_circles)
          nvds_add_display_meta_to_frame (frame_meta, disp_meta);
    }
    return GST_PAD_PROBE_OK;
  }
//****************************************************************************************************************

/*Generate ROS messages right after display meta */
  static GstPadProbeReturn
    ros_publish_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
        gpointer u_data)
    {
    GstBuffer *buf = (GstBuffer *) info->data;
    NvDsObjectMeta *obj_meta = NULL;
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    int part_index = 0;
    geometry_msgs::TransformStamped gaze_transform;
    geometry_msgs::TransformStamped head_transform;
    tf2::Quaternion q_aux;
    
    PerceptionStream *perception = (PerceptionStream *)(u_data);

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
        l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);    
        NvDsDisplayMeta *disp_meta = NULL;
 
        if (!frame_meta)
        continue;

        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
         l_obj = l_obj->next) {
        obj_meta = (NvDsObjectMeta *) (l_obj->data);

        if (!obj_meta)
            continue;

        hri_msgs::NormalizedPointOfInterest2D p; //NormalizedPointOfInterest2D
                       
        for (NvDsMetaList * l_user = obj_meta->obj_user_meta_list;
            l_user != NULL; l_user = l_user->next) {
            NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
            if(user_meta->base_meta.meta_type ==
                (NvDsMetaType)NVDS_USER_RIVA_META_FACEMARK) {
            NvDsFacePointsMetaData *facepoints_meta =
                (NvDsFacePointsMetaData *)user_meta->user_meta_data;
          /*Get the face marks and mark with dots*/
              if (!facepoints_meta)
                continue;
              for (part_index = 0;part_index < facepoints_meta->facemark_num;
                part_index++) {  
                p.x = facepoints_meta->mark[part_index].x;
            	  p.y = facepoints_meta->mark[part_index].y;
            	  p.c = facepoints_meta->mark[part_index].score;
            	  perception->FacialLandmarks.landmarks.push_back(p);
              }
            } else if(user_meta->base_meta.meta_type ==
                (NvDsMetaType)NVDS_USER_RIVA_META_GAZE) {
                NvDsGazeMetaData * gazemeta =
                    (NvDsGazeMetaData *)user_meta->user_meta_data;
                
                q_aux.setRPY(0,gazemeta->gaze_params[4],gazemeta->gaze_params[3]);
                q_aux.normalize();
            }
         }
         //perception->emotion.data = obj_meta->obj_label;
         //perception->h_rate.data = obj_meta->misc_obj_info[0];
      
      geometry_msgs::PoseStamped face;
      const auto head_orientation = toQuaternion(perception->head_pose[4], -perception->head_pose[3], -perception->head_pose[5]);
	    face.pose.orientation = toQuaternion(M_PI,  0,  0);//toQuaternion(M_PI / 2, 0, M_PI / 2);// toQuaternion(0, 0, 0);
	    face.pose.orientation = face.pose.orientation * head_orientation;
	    // tf
	    gaze_transform.header.frame_id = perception->face_frame_id;
	    gaze_transform.header.stamp = ros::Time::now();
	    gaze_transform.child_frame_id = "child_gaze";
	    gaze_transform.transform.translation.x = 0;
	    gaze_transform.transform.translation.y = 0;
	    gaze_transform.transform.translation.z = 0;
	    gaze_transform.transform.rotation = tf2::toMsg(q_aux);
	  
	    head_transform.header.frame_id = perception->camera_frame_id;
	    head_transform.header.stamp = ros::Time::now();
	    head_transform.child_frame_id = perception->face_frame_id;
	    head_transform.transform.translation.x = perception->head_pose[0] / 1000.0;
	    head_transform.transform.translation.y = perception->head_pose[1] / 1000.0;
	    head_transform.transform.translation.z = perception->head_pose[2] / 1000.0;
	    head_transform.transform.rotation = face.pose.orientation;
	  
	    perception->face_transform.clear();
	    perception->face_transform.push_back(head_transform);
	    perception->face_transform.push_back(gaze_transform);

      g_print ("Tx = %f Ty = %f Tz = %f Rx = %f Ry = %f Rz = %f\n" , perception->head_pose[0], perception->head_pose[1], perception->head_pose[2],
      perception->head_pose[3], perception->head_pose[4], perception->head_pose[5]);
      }
      if (disp_meta && disp_meta->num_circles)
        nvds_add_display_meta_to_frame (frame_meta, disp_meta);
    }
    return GST_PAD_PROBE_OK;
  }

//****************************************************************************************************************
/* This is the buffer probe function that we have registered on the src pad
 * of the PGIE's next queue element. The face bbox will be scale to square for
 * facial marks.
 */
  static GstPadProbeReturn
  pgie_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
  {
  NvDsBatchMeta *batch_meta =
      gst_buffer_get_nvds_batch_meta (GST_BUFFER (info->data));
  NvBufSurface *in_surf;
  GstMapInfo in_map_info;
  int frame_width, frame_height;

  memset (&in_map_info, 0, sizeof (in_map_info));

  /* Map the buffer contents and get the pointer to NvBufSurface. */
  if (!gst_buffer_map (GST_BUFFER (info->data), &in_map_info, GST_MAP_READ)) {
    g_printerr ("Failed to map GstBuffer\n");
    return GST_PAD_PROBE_PASS;
  }
  in_surf = (NvBufSurface *) in_map_info.data;
  gst_buffer_unmap(GST_BUFFER (info->data), &in_map_info);

  /* Iterate each frame metadata in batch */
  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
    frame_width = in_surf->surfaceList[frame_meta->batch_id].width;
    frame_height = in_surf->surfaceList[frame_meta->batch_id].height;
    NvDsDisplayMeta *disp_meta = NULL;
    disp_meta = nvds_acquire_display_meta_from_pool(batch_meta);
    disp_meta->num_circles = 0;
    disp_meta->num_rects = 0;
    disp_meta->num_labels = 0;
        
    disp_meta->rect_params[disp_meta->num_rects].left = frame_width/2 - 60;
    disp_meta->rect_params[disp_meta->num_rects].top = frame_height/2 - 60;
    disp_meta->rect_params[disp_meta->num_rects].width =120;
    disp_meta->rect_params[disp_meta->num_rects].height =120;
    disp_meta->rect_params[disp_meta->num_rects].border_width = 2;
    disp_meta->rect_params[disp_meta->num_rects].border_color.red = 0.0;
    disp_meta->rect_params[disp_meta->num_rects].border_color.green = 0.0;
    disp_meta->rect_params[disp_meta->num_rects].border_color.blue = 1.0;
    disp_meta->rect_params[disp_meta->num_rects].border_color.alpha = 0.5;
    disp_meta->num_rects+=1;
    /* Iterate object metadata in frame */
    for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL;
        l_obj = l_obj->next) {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) l_obj->data;
      float center_x = obj_meta->rect_params.width/2.0 +
          obj_meta->rect_params.left;
      float center_y = obj_meta->rect_params.height/2.0 +
          obj_meta->rect_params.top;
      if (sqrt(pow(frame_width/2 - center_x, 2.0) + pow(frame_height/2 - center_y, 2.0)) > 100){ 
        nvds_remove_obj_meta_from_frame(frame_meta, obj_meta);
      }
      else{
        if (!obj_meta) {
          g_print("No obj meta\n");
          break;
        }
        
        if(obj_meta->rect_params.left<0)
          obj_meta->rect_params.left=0;
        if(obj_meta->rect_params.top<0)
          obj_meta->rect_params.top=0;
          
        float square_size = MAX(obj_meta->rect_params.width,
          obj_meta->rect_params.height) * 1.2;
        float center_x = obj_meta->rect_params.width/2.0 +
          obj_meta->rect_params.left;
        float center_y = obj_meta->rect_params.height/2.0 +
          obj_meta->rect_params.top;

        /*Check the border*/
        if(center_x < (square_size/2.0) || center_y < square_size/2.0 || 
          center_x + square_size/2.0 > frame_width ||
          center_y - square_size/2.0 > frame_height) {
          g_print("Keep the original bbox\n");
        } else {
          obj_meta->rect_params.left = center_x - square_size/2.0;
          obj_meta->rect_params.top = center_y - square_size/2.0;
          obj_meta->rect_params.width = square_size;
          obj_meta->rect_params.height = square_size;
        }
      }
    }
    if (disp_meta && disp_meta->num_rects)
          nvds_add_display_meta_to_frame (frame_meta, disp_meta);   
  }
  return GST_PAD_PROBE_OK;
  }

/* This is the buffer probe function that we have registered on the src pad
 * of the SGIE's next queue element. The facial marks output will be processed
 * and the facial marks metadata will be generated.
 */
  static GstPadProbeReturn
  sgie_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
  {
    NvDsBatchMeta *batch_meta =
      gst_buffer_get_nvds_batch_meta (GST_BUFFER (info->data));
    PerceptionStream *perception = (PerceptionStream *)(u_data);
    /* Iterate each frame metadata in batch */
    for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
      NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
      //NvDsDisplayMeta *disp_meta = NULL;
      /* Iterate object metadata in frame */
      for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL;
        l_obj = l_obj->next) {

        NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) l_obj->data;

        if (!obj_meta)
          continue;

        /* Iterate user metadata in object to search SGIE's tensor data */
        for (NvDsMetaList * l_user = obj_meta->obj_user_meta_list; l_user != NULL;
          l_user = l_user->next) {
          NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
          if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
            continue;

          NvDsInferTensorMeta *meta =
            (NvDsInferTensorMeta *) user_meta->user_meta_data;
          float * heatmap_data = NULL;
          float * confidence = NULL;
          //int heatmap_c = 0;

          for (unsigned int i = 0; i < meta->num_output_layers; i++) {
            NvDsInferLayerInfo *info = &meta->output_layers_info[i];
            info->buffer = meta->out_buf_ptrs_host[i];

            std::vector < NvDsInferLayerInfo >
            outputLayersInfo (meta->output_layers_info,
            meta->output_layers_info + meta->num_output_layers);
            //Prepare CVCORE input layers
            if (strcmp(outputLayersInfo[i].layerName,
              "output_1") == 0) {
              //This layer output landmarks coordinates
              heatmap_data = (float *)meta->out_buf_ptrs_host[i];
            } else if (strcmp(outputLayersInfo[i].layerName,
              "softargmax:1") == 0) {
              confidence = (float *)meta->out_buf_ptrs_host[i];
            }
          }
        
          int heatmap_h = 2;
          int heatmap_c = 106;
          float confi[perception->n_part_points] = { };
        
          cvcore::Array<cvcore::ArrayN<cvcore::Vector2f,
            cvcore::faciallandmarks::FacialLandmarks::MAX_NUM_FACIAL_LANDMARKS>>
          output(1, true);
          output.setSize(1);
          int j=0;              
          for (int i = 0; i < perception->mappings.size(); i++)
          {
                j = perception->mappings[i];
                float land_x = heatmap_data[j * heatmap_h];
                float land_y = heatmap_data[j * heatmap_h + 1];
                confi[i] = 0.5;
                land_x = land_x * obj_meta->rect_params.width;
                land_y = land_y * obj_meta->rect_params.height; 
                output[0][i].x=land_x;
                output[0][i].y=land_y;
                //g_print ("X = %f Y = %f\n", land_x, land_y);
          }
        
          confidence =(float *) confi;
          /*add user meta for facemark*/
          if (!nvds_add_facemark_meta (batch_meta, obj_meta, output[0],
            confidence)) {
            g_printerr ("Failed to get bbox from model output\n");
          }
        }
      }
    }
    return GST_PAD_PROBE_OK;
  }

  static gboolean
  bus_call (GstBus * bus, GstMessage * msg, gpointer data)
    {
    GMainLoop *loop = (GMainLoop *) data;
    switch (GST_MESSAGE_TYPE (msg)) {
        case GST_MESSAGE_EOS:
        g_print ("End of stream\n");
        g_main_loop_quit (loop);
        break;
        case GST_MESSAGE_ERROR:{
        gchar *debug;
        GError *error;
        gst_message_parse_error (msg, &error, &debug);
        g_printerr ("ERROR from element %s: %s\n",
              GST_OBJECT_NAME (msg->src), error->message);
        if (debug)
            g_printerr ("Error details: %s\n", debug);
        g_free (debug);
        g_error_free (error);
        g_main_loop_quit (loop);
        break;
        }
        default:
          break;
    }
    return TRUE;
  }
  
static void
cb_newpad (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data)
{
  g_print ("In cb_newpad\n");
  GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  PerceptionStream *perception = (PerceptionStream *)(data);
  GstCapsFeatures *features = gst_caps_get_features (caps, 0);

  /* Need to check if the pad created by the decodebin is for video and not
   * audio. */
  if (!strncmp (name, "video", 5)) {
    /* Link the decodebin pad to videoconvert if no hardware decoder is used */
    if (perception->source_struct.vidconv) {
      GstPad *conv_sink_pad = gst_element_get_static_pad (perception->source_struct.vidconv,
          "sink");
      if (gst_pad_link (decoder_src_pad, conv_sink_pad)) {
        g_printerr ("Failed to link decoderbin src pad to"
            " converter sink pad\n");
      }
      g_object_unref(conv_sink_pad);
      if (!gst_element_link (perception->source_struct.vidconv, perception->source_struct.nvvidconv)) {
         g_printerr ("Failed to link videoconvert to nvvideoconvert\n");
      }
    } else {
      GstPad *conv_sink_pad = gst_element_get_static_pad (perception->source_struct.nvvidconv,
          "sink");
      if (gst_pad_link (decoder_src_pad, conv_sink_pad)) {
        g_printerr ("Failed to link decoderbin src pad to "
            "converter sink pad\n");
      }
      g_object_unref(conv_sink_pad);
    }
    if (gst_caps_features_contains (features, GST_CAPS_FEATURES_NVMM)) {
      g_print ("###Decodebin pick nvidia decoder plugin.\n");
    } else {
      /* Get the source bin ghost pad */
      g_print ("###Decodebin did not pick nvidia decoder plugin.\n");
    }
  }
}

static void
decodebin_child_added (GstChildProxy * child_proxy, GObject * object,
    gchar * name, gpointer user_data)
{
  PerceptionStream *perception = (PerceptionStream *)(user_data);
  g_print ("Decodebin child added: %s\n", name);
  if (g_strrstr (name, "decodebin") == name) {
    g_signal_connect (G_OBJECT (object), "child-added",
        G_CALLBACK (decodebin_child_added), user_data);
  }
  if (g_strstr_len (name, -1, "pngdec") == name) {
    perception->source_struct.vidconv = gst_element_factory_make ("videoconvert",
        "source_vidconv");
    gst_bin_add (GST_BIN (perception->source_struct.source_bin), perception->source_struct.vidconv);
  } else {
    perception->source_struct.vidconv = NULL;
  }
}

static bool
create_source_bin (gpointer user_data, const char * uri)
{
  PerceptionStream *perception = (PerceptionStream *)(user_data);
  gchar bin_name[16] = { };
  GstCaps *caps = NULL;
  GstCapsFeatures *feature = NULL;

  perception->source_struct.nvvidconv = NULL;
  perception->source_struct.capsfilt = NULL;
  perception->source_struct.source_bin = NULL;
  perception->source_struct.uri_decode_bin = NULL;

  g_snprintf (bin_name, 15, "source-bin-%02d", perception->source_struct.index);
  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  perception->source_struct.source_bin = gst_bin_new (bin_name);

  /* Source element for reading from the uri.
   * We will use decodebin and let it figure out the container format of the
   * stream and the codec and plug the appropriate demux and decode plugins. */
  perception->source_struct.uri_decode_bin = gst_element_factory_make ("uridecodebin",
      "uri-decode-bin");
  perception->source_struct.nvvidconv = gst_element_factory_make ("nvvideoconvert",
      "source_nvvidconv");
  perception->source_struct.capsfilt = gst_element_factory_make ("capsfilter",
      "source_capset");

  if (!perception->source_struct.source_bin || !perception->source_struct.uri_decode_bin ||
      !perception->source_struct.nvvidconv
      || !perception->source_struct.capsfilt) {
    g_printerr ("One element in source bin could not be created.\n");
    return false;
  }

  /* We set the input uri to the source element */
  g_object_set (G_OBJECT (perception->source_struct.uri_decode_bin), "uri", uri, NULL);

  /* Connect to the "pad-added" signal of the decodebin which generates a
   * callback once a new pad for raw data has beed created by the decodebin */
  g_signal_connect (G_OBJECT (perception->source_struct.uri_decode_bin), "pad-added",
      G_CALLBACK (cb_newpad), user_data);
  g_signal_connect (G_OBJECT (perception->source_struct.uri_decode_bin), "child-added",
      G_CALLBACK (decodebin_child_added), user_data);

  caps = gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING, "NV12",
      NULL);
  feature = gst_caps_features_new ("memory:NVMM", NULL);
  gst_caps_set_features (caps, 0, feature);
  g_object_set (G_OBJECT (perception->source_struct.capsfilt), "caps", caps, NULL);

#ifndef PLATFORM_TEGRA
  g_object_set (G_OBJECT (perception->source_struct.nvvidconv), "nvbuf-memory-type", 3,
      NULL);
#endif

  gst_bin_add_many (GST_BIN (perception->source_struct.source_bin),
      perception->source_struct.uri_decode_bin, perception->source_struct.nvvidconv,
      perception->source_struct.capsfilt, NULL);

  if (!gst_element_link (perception->source_struct.nvvidconv,
      perception->source_struct.capsfilt)) {
    g_printerr ("Could not link vidconv and capsfilter\n");
    return false;
  }

  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  GstPad *gstpad = gst_element_get_static_pad (perception->source_struct.capsfilt,
      "src");
  if (!gstpad) {
    g_printerr ("Could not find srcpad in '%s'",
        GST_ELEMENT_NAME(perception->source_struct.capsfilt));
      return false;
  }
  if(!gst_element_add_pad (perception->source_struct.source_bin,
      gst_ghost_pad_new ("src", gstpad))) {
    g_printerr ("Could not add ghost pad in '%s'",
        GST_ELEMENT_NAME(perception->source_struct.capsfilt));
  }
  gst_object_unref (gstpad);

  return true;
}  
/***************************************************************************************/  
  PerceptionStream::PerceptionStream(ros::NodeHandle nodeh) :
    pipeline_(NULL),
    nh(nodeh),
    image_transport(nodeh)
  {
  }
/***************************************************************************************/
  PerceptionStream::~PerceptionStream()
  {
  }
/***************************************************************************************/
  bool PerceptionStream::configure()
  {
      /* Read cvcore parameters from config file.*/
    int numFaciallandmarks=80;
    int maxBatchSize=32;
    int inputLayerWidth=80;
    int inputLayerHeight=80;
    
    cvcore::ModelInputParams ModelInputParams = {32, 80, 80, cvcore::Y_F32};
    
    nh.param("/ros_hz", ros_hz,10);
    nh.param<std::string>("/camera_frame_id", camera_frame_id, "zed2_camera_center");
    nh.param<std::string>("/face_frame_id", face_frame_id, "face_frame");
    nh.param<std::string>("/world_frame_id", world_frame_id, "world");

    nh.param<std::string>("/configs_path", configs_path, "home");
       
    config_primary = configs_path + "/configs/config_infer_primary_face_retina.txt";
    config_secondary = configs_path + "/configs/face_sdk_sgie_config.txt";
    config_gaze = "config-file:" + configs_path + "/configs/sample_gazenet_model_config.txt";
    config_tracker = configs_path + "/configs/config_tracker_NvDCF_max_perf.yml";
    
    gaze_lib = configs_path + "/apps/libnvds_gazeinfer.so";
    tracker_lib = "/opt/nvidia/deepstream/deepstream-6.0/lib/libnvds_nvmultiobjecttracker.so";
    
    nh.param("/os_display", os_display, true);
    nh.param("/src_zed_gst", src_zed_gst, false);
    
    frame_width=736;
    frame_height=416;
                
    nh.param("/zed_resolution", zed_resolution, 1);
    
    switch (zed_resolution)  {
        case 1:
            stream_width=1920;
            stream_height=1080;
            fx = 1095.98583984375;
            fy = 1095.98583984375;
            cx = 942.5280151367188;
            cy = 543.7188720703125;
            break;

        case 2:
            stream_width=1280;
            stream_height=720;
            fx = 539.65234375;
            fy = 539.65234375;
            cx = 622.3931884765625;
            cy = 361.3262939453125;
            break;
        case 3:
            stream_width=672;//640;
            stream_height=376;//480;
            fx = 537.24;
            fy = fx;
            cx = 409;
            cy = 209;
            break; 
        default:
            stream_width=640;
            stream_height=480;
            fx = 537.24;
            fy = fx;
            cx = stream_width/2;
            cy = stream_height/2;
            break;
    }
    
  
    ModelInputParams.maxBatchSize = (size_t)maxBatchSize;
    ModelInputParams.inputLayerWidth = (size_t)inputLayerWidth;
    ModelInputParams.inputLayerHeight = (size_t)inputLayerHeight;
    
    mappings = {0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,33,34,35,36,37,42,43,44,45,46,51,52,53,54,57,
          59,60,61,63,66,67,69,70,71,72,75,76,78,79,80,82,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103};
    n_part_points = mappings.size();
       
    old_landmarks.create(2 * n_part_points, 1) ;
    
    rotation_hypotheses.push_back(cv::Vec3d(0, 1.57, 0));
    rotation_hypotheses.push_back(cv::Vec3d(0,0,0));
    rotation_hypotheses.push_back(cv::Vec3d(0, -0.5236, 0));
    rotation_hypotheses.push_back(cv::Vec3d(0, 0.5236,0));
    rotation_hypotheses.push_back(cv::Vec3d(0, -0.96, 0));
    rotation_hypotheses.push_back(cv::Vec3d(0, 0.96, 0));
    rotation_hypotheses.push_back(cv::Vec3d(0, 0, 0.5236));
    rotation_hypotheses.push_back(cv::Vec3d(0, 0, -0.5236));
    rotation_hypotheses.push_back(cv::Vec3d(0, -1.57, 0));
    rotation_hypotheses.push_back(cv::Vec3d(0, 1.22, -0.698));
    rotation_hypotheses.push_back(cv::Vec3d(0, -1.22, 0.698));
    
    std::unique_ptr< cvcore::faciallandmarks::FacialLandmarksPostProcessor >    faciallandmarkpostinit(
        new cvcore::faciallandmarks::FacialLandmarksPostProcessor (
        ModelInputParams,numFaciallandmarks));
    facemarkpost = std::move(faciallandmarkpostinit);
    ROS_INFO("Config Done.");
    return true;
  }  
/***************************************************************************************/
  bool PerceptionStream::init_stream()
  {
    
    GstElement *streammux = NULL, *sink = NULL, 
                 *primary_detector = NULL, *second_detector = NULL,
                 *nvvidconv = NULL, *nvosd = NULL, *nvvidconv1 = NULL,
                 *outenc = NULL, *capfilt = NULL,
                 *gaze_identifier = NULL, *hrinfer = NULL;
    GstElement *source_bin = NULL, *zedsrc = NULL, *videoconvert = NULL, *tracker = NULL,
                *src_nvvidconv = NULL, *src_capsfilt = NULL, *v4l_capsfilt = NULL;
    GstElement *queue1 = NULL, *queue2 = NULL, *queue3 = NULL, *queue4 = NULL,
               *queue5 = NULL, *queue6 = NULL, *queue7 = NULL, *queue8 = NULL;
    
#ifdef PLATFORM_TEGRA
    GstElement *transform = NULL;
#endif
    GstBus *bus = NULL;
    GstPad *osd_sink_pad = NULL;
    GstCaps *caps = NULL;
    GstCapsFeatures *feature = NULL;
    
    GstCaps *src_caps = NULL, *v4lsrc_caps = NULL;
    GstCapsFeatures *src_feature = NULL;
    //int i;
    src_cnt = 1;
    guint tiler_rows, tiler_columns;
        
    GstPad *sinkpad, *srcpad;
    gchar pad_name_sink[16] = "sink_0";
    gchar pad_name_src[16] = "src";
    gchar *filename;
    ifstream fconfig;
    
    /* Standard GStreamer initialization */
    if(!gst_is_initialized()){
    ROS_INFO("Initializing gstreamer...");
    gst_init (NULL, NULL);
    }
    
    loop = g_main_loop_new (NULL, FALSE);
  
    perf_measure.pre_time = GST_CLOCK_TIME_NONE;
    perf_measure.total_time = GST_CLOCK_TIME_NONE;
    perf_measure.count = 0;  

    /* Create gstreamer elements */
    /* Create Pipeline element that will form a connection of other elements */
    pipeline_ = gst_pipeline_new ("pipeline"); 

    /* Create nvstreammux instance to form batches from one or more sources. */
    streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

    if (!pipeline_ || !streammux) {
          g_printerr ("One main element could not be created. Exiting.\n");
          return -1;
    }

    gst_bin_add (GST_BIN(pipeline_), streammux);
    
    /* Source element: reading from the file; Zed gst plugin or v4l2src */
    
    source_bin = gst_bin_new ("source-bin-01");
    
    if(!src_zed_gst){
      zedsrc = gst_element_factory_make ("v4l2src", "src_elem");
    } else{
      zedsrc = gst_element_factory_make ("zedsrc", "source_zedsrc");
    }
  
    videoconvert = gst_element_factory_make ("videoconvert", "source_videoconvert");    
    src_nvvidconv = gst_element_factory_make ("nvvideoconvert", "source_nvvidconv");
    src_capsfilt = gst_element_factory_make ("capsfilter", "source_capset");
    v4l_capsfilt = gst_element_factory_make ("capsfilter", "v4lsource_capset");
    
    if (!source_bin || !zedsrc || !videoconvert || !src_nvvidconv || !src_capsfilt || !v4l_capsfilt) {
      g_printerr ("One element in source bin could not be created.\n");
      return false;
    }
  
    src_caps = gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING, "NV12", NULL);
    src_feature = gst_caps_features_new ("memory:NVMM", NULL);
    gst_caps_set_features (src_caps, 0, src_feature);
    g_object_set (G_OBJECT (src_capsfilt), "caps", src_caps, NULL);
  
    v4lsrc_caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "YUY2", 
	  "width", G_TYPE_INT, stream_width*2, "height", G_TYPE_INT, stream_height, 
	  "framerate", GST_TYPE_FRACTION, 60, 1, NULL);
    g_object_set (G_OBJECT (v4l_capsfilt), "caps", v4lsrc_caps, NULL);
  
    guint left=(stream_width-frame_width)/2; //Para formar el src crop="left:top:width:height"
    guint top=(stream_height-frame_height)/2;
    stringstream crop;
    crop << left <<":"<< top << ":" << frame_width << ":" << frame_height;

    g_object_set (G_OBJECT (src_nvvidconv), "src-crop", crop.str().c_str() , NULL);//crop.str() "272:152:736:416"
  
    if(!src_zed_gst){
      g_object_set (G_OBJECT (zedsrc), "device", "/dev/video0", NULL);
      gst_bin_add_many (GST_BIN (source_bin),zedsrc,v4l_capsfilt,src_nvvidconv,src_capsfilt, NULL);
      
    if (!gst_element_link_many (zedsrc, v4l_capsfilt, src_nvvidconv, src_capsfilt, NULL)) {
          g_printerr ("Could not link zedsrc, vidconvert, nvvidconv and capsfilter\n");
          return false;
      }
  } else{
     g_object_set (G_OBJECT (zedsrc), "stream-type", 0, "camera-resolution", zed_resolution, "camera-fps", 30, "enable-positional-tracking", false, "depth-mode", 0, "depth-stabilization", false, "od-enabled", false,"od-enable-tracking", false, "aec-agc" ,false, "set-as-static", true, NULL);
     gst_bin_add_many (GST_BIN (source_bin),zedsrc,videoconvert,src_nvvidconv,src_capsfilt, NULL);
      
     if (!gst_element_link_many (zedsrc, videoconvert, src_nvvidconv, src_capsfilt, NULL)) {
        g_printerr ("Could not link zedsrc, vidconvert, nvvidconv and capsfilter\n");
        return false;
      }
    }
    
    //*********

    GstPad *gstpad = gst_element_get_static_pad (src_capsfilt,"src");
    if (!gstpad) {
      g_printerr ("Could not find srcpad in '%s'", GST_ELEMENT_NAME(src_capsfilt));
      return false;
    }
    if(!gst_element_add_pad (source_bin, gst_ghost_pad_new ("src", gstpad))) {
      g_printerr ("Could not add ghost pad in '%s'", GST_ELEMENT_NAME(src_capsfilt));
    }
    gst_object_unref (gstpad);
    //*********  
    
    gst_bin_add (GST_BIN (pipeline_), source_bin);
    srcpad = gst_element_get_static_pad (source_bin, pad_name_src);
    if (!srcpad) {
      g_printerr ("Decoder zedsrc request src pad failed. Exiting.\n");
      return -1;
    }
    sinkpad = gst_element_get_request_pad (streammux, pad_name_sink);
     
    if (!sinkpad) {
      g_printerr ("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link decoder to stream muxer. Exiting.\n");
      return -1;
    }
    
    gst_object_unref (sinkpad);
    gst_object_unref (srcpad);
    
    /* Create three nvinfer instances for two detectors and one classifier*/
    primary_detector = gst_element_factory_make ("nvinfer", "primary-infer-engine1");

    second_detector = gst_element_factory_make ("nvinfer", "second-infer-engine1");

    /* Use convertor to convert from NV12 to RGBA as required by nvosd */
    nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvid-converter");

    /* Create OSD to draw on the converted RGBA buffer */
    nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

    nvvidconv1 = gst_element_factory_make ("nvvideoconvert", "nvvid-converter1");

    capfilt = gst_element_factory_make ("capsfilter", "nvvideo-caps");

    gaze_identifier = gst_element_factory_make ("nvdsvideotemplate", "gaze_infer");
    
    tracker = gst_element_factory_make ("nvtracker", "tracking_tracker");
    
    g_object_set (G_OBJECT (tracker), "tracker-width", frame_width, "tracker-height", frame_height, "ll-config-file", config_tracker.c_str(), "ll-lib-file", tracker_lib.c_str(), NULL);
 
    queue1 = gst_element_factory_make ("queue", "queue1");
    queue2 = gst_element_factory_make ("queue", "queue2");
    queue3 = gst_element_factory_make ("queue", "queue3");
    queue4 = gst_element_factory_make ("queue", "queue4");
    queue5 = gst_element_factory_make ("queue", "queue5");
    queue6 = gst_element_factory_make ("queue", "queue6");
    queue7 = gst_element_factory_make ("queue", "queue7");
    queue8 = gst_element_factory_make ("queue", "queue8");  
    
    if(os_display){
#ifdef PLATFORM_TEGRA
    transform = gst_element_factory_make ("nvegltransform", "nvegltransform");
    if(!transform) {
      g_printerr ("nvegltransform element could not be created. Exiting.\n");
      return -1;
    }
#endif
    sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
    } else{
    sink = gst_element_factory_make ("fakesink", "fake-renderer");
    }
    
    if (!primary_detector || !second_detector || !nvvidconv || !nvosd || !sink  || !capfilt || !gaze_identifier) {
        g_printerr ("One element could not be created. Exiting.\n");
        return -1;
    }

    g_object_set (G_OBJECT (streammux), "width", frame_width, "height",
      frame_height, "batch-size", src_cnt,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, "live-source", 1 , NULL);
     
    g_object_set (G_OBJECT (primary_detector), "config-file-path", config_primary.c_str(), "unique-id", PRIMARY_DETECTOR_UID, NULL);

    g_object_set (G_OBJECT (second_detector), "config-file-path", config_secondary.c_str(), "unique-id", SECOND_DETECTOR_UID, NULL);
     
    g_object_set (G_OBJECT (gaze_identifier), "customlib-name", gaze_lib.c_str(), "customlib-props", config_gaze.c_str(), NULL);
    
    /* we add a bus message handler */
     bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline_));
     bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
     gst_object_unref (bus);

    /* Set up the pipeline */
    /* we add all elements into the pipeline */
    caps = gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING, "RGBA", NULL);
    feature = gst_caps_features_new ("memory:NVMM", NULL);
    gst_caps_set_features (caps, 0, feature);
    g_object_set (G_OBJECT (capfilt), "caps", caps, NULL);
    
    gst_bin_add_many (GST_BIN (pipeline_), primary_detector, tracker, second_detector,
      queue1, queue2, queue4, queue5, nvvidconv, capfilt, nvosd, sink,
      gaze_identifier, queue7, NULL); //hrinfer, emotioninfer, queue3, queue6,

    if (!gst_element_link_many (streammux, queue1, primary_detector, tracker, queue2, 
          second_detector, queue4, gaze_identifier, queue5,//hrinfer, queue3,
          nvvidconv, capfilt, queue7, nvosd, NULL)) {//emotioninfer, queue6, 
     g_printerr ("Inferring and tracking elements link failure.\n");
     return -1;
     }

     g_object_set (G_OBJECT (sink), "sync", 0, "async", false,NULL);
   
    if(os_display){  
#ifdef PLATFORM_TEGRA
     gst_bin_add_many (GST_BIN (pipeline_), transform, queue8, NULL);
     if (!gst_element_link_many (nvosd, queue8, transform, sink, NULL)) {
        g_printerr ("OSD and sink elements link failure.\n");
        return -1;
    }
#else
    gst_bin_add (GST_BIN (pipeline_), queue8);
    if (!gst_element_link_many (nvosd, queue8, sink, NULL)) {
        g_printerr ("OSD and sink elements link failure.\n");
        return -1;
    }
#endif    
    } else{
        if (!gst_element_link(nvosd, sink)) {
            g_printerr ("OSD and sink elements link failure.\n");
            return -1;
        }
    }
    /* Display the facemarks output on video. Fakesink do not need to display. */
    osd_sink_pad = gst_element_get_static_pad (queue5, "src");
    if (!osd_sink_pad)
        g_print ("Unable to get sink pad\n");
    else
        gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
          tile_sink_pad_buffer_probe, this, NULL);
    gst_object_unref (osd_sink_pad);
    
    /* publish the facemarks and output data on ROS env*/
    osd_sink_pad = gst_element_get_static_pad (queue5, "sink");
    if (!osd_sink_pad)
        g_print ("Unable to get sink pad\n");
    else
        gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
          ros_publish_sink_pad_buffer_probe, this, NULL);
    gst_object_unref (osd_sink_pad);
        
    /*Performance measurement*/
    osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
    if (!osd_sink_pad)
        g_print ("Unable to get sink pad\n");
    else
        gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
            osd_sink_pad_buffer_probe, this, NULL);
    gst_object_unref (osd_sink_pad);

    /* Add probe to get square bbox from face detector. */
    osd_sink_pad = gst_element_get_static_pad (queue2, "src");
    if (!osd_sink_pad)
        g_print ("Unable to get nvinfer src pad\n");
    gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        pgie_pad_buffer_probe, NULL, NULL);
    gst_object_unref (osd_sink_pad);

    /* Add probe to handle facial marks output and generate facial */
    /* marks metadata.                                             */
    osd_sink_pad = gst_element_get_static_pad (queue4, "src");
    if (!osd_sink_pad)
        g_print ("Unable to get nvinfer src pad\n");
    gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        sgie_pad_buffer_probe, this, NULL);
    gst_object_unref (osd_sink_pad);
    
    gst_element_set_state (pipeline_, GST_STATE_PAUSED);
    
    if(gst_element_get_state (pipeline_, NULL, NULL, -1) == GST_STATE_CHANGE_FAILURE){
    ROS_ERROR("Failed to PAUSE stream");
    return false;
    }

    ROS_INFO("gstreamer initialised and pipeline is PAUSED");
    return true;
  }
/***************************************************************************************/
  void PerceptionStream::publish_stream()
  {
    
    ros::Rate r(ros_hz); // 6 hz
    /* Set the pipeline to "playing" state */  
    gst_element_set_state (pipeline_, GST_STATE_PLAYING);
    if(gst_element_get_state (pipeline_, NULL, NULL, -1) == GST_STATE_CHANGE_FAILURE){
    ROS_ERROR("Failed to PLAY during prerroll");
    return;
    }
    gst_element_set_state (pipeline_, GST_STATE_PAUSED);
    if(gst_element_get_state (pipeline_, NULL, NULL, -1) == GST_STATE_CHANGE_FAILURE){
    ROS_ERROR("Failed to PAUSE during prerroll");
    return;
    }
    
    if(gst_element_set_state (pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE){
    ROS_ERROR("Failed to PLAY");
    return;
    }
    ROS_INFO("Started Stream");
    while(ros::ok()) 
    {
      tf_br.sendTransform(face_transform);
      ros::spinOnce();
      r.sleep();
    }
    /* Out of the main loop, clean up nicely */
    g_print ("Returned, stopping playback\n");
    
  }
/***************************************************************************************/
  void PerceptionStream::cleanup_stream()
  {
    gst_element_set_state (pipeline_, GST_STATE_NULL);
    g_print (" ==== Tobo_Perception_Node ended ===== \n");
    if(perf_measure.total_time)
    {
        g_print ("-- Average fps: %f\n",
          ((perf_measure.count-1)*src_cnt*1000000.0)/perf_measure.total_time);
    }
    
    g_print ("-- Totally frames adquired: %d \n",frame_number);
    g_print ("-- Totally faces inferred: %d \n",total_face_num);
    g_print ("-- Percentage Faces detected: %f %% \n", total_face_num*100.0/frame_number);
    g_print ("Deleting pipeline\n");
    gst_object_unref (GST_OBJECT (pipeline_));
    g_source_remove (bus_watch_id);
  }
/**************************************************************************************/
  void PerceptionStream::run() {
    while(ros::ok()) {
      if(!this->configure()) {
        ROS_FATAL("Failed to configure gscam!");
        break;
      }

      if(!this->init_stream()) {
        ROS_FATAL("Failed to initialize gscam stream!");
        break;
      }

      // Block while publishing
      this->publish_stream();

      this->cleanup_stream();

      g_print("GStreamer stream stopped!");
      g_print("Cleaning up stream and exiting...");
      break;
    }

  }
/**************************************************************************************/

}

