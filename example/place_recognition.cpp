#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <boost/filesystem.hpp>

#include "include/btc.h"
#include "include/utils.h"

// Read KITTI data
std::vector<float> read_lidar_data(const std::string lidar_data_path) {
  std::ifstream lidar_data_file;
  lidar_data_file.open(lidar_data_path,
                       std::ifstream::in | std::ifstream::binary);
  if (!lidar_data_file) {
    std::cout << "Read End..." << std::endl;
    std::vector<float> nan_data;
    return nan_data;
    // exit(-1);
  }
  lidar_data_file.seekg(0, std::ios::end);
  const size_t num_elements = lidar_data_file.tellg() / sizeof(float);
  lidar_data_file.seekg(0, std::ios::beg);

  std::vector<float> lidar_data_buffer(num_elements);
  lidar_data_file.read(reinterpret_cast<char *>(&lidar_data_buffer[0]),
                       num_elements * sizeof(float));
  return lidar_data_buffer;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "btc_place_recognition");
  ros::NodeHandle nh;
  std::string setting_path = "";
  std::string pcds_dir = "";
  std::string pose_file = "";
  std::string result_file = "";
  double cloud_overlap_thr = 0.5;
  bool calc_gt_enable = false;
  bool read_bin = true;
  nh.param<double>("cloud_overlap_thr", cloud_overlap_thr, 0.5);
  nh.param<std::string>("setting_path", setting_path, "");
  nh.param<std::string>("pcds_dir", pcds_dir, "");
  nh.param<std::string>("pose_file", pose_file, "");
  nh.param<bool>("read_bin", read_bin, true);

  ros::Publisher pubOdomAftMapped =
      nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
  ros::Publisher pubCureentCloud =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_current", 100);
  ros::Publisher pubCurrentBinary =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_key_points", 100);
  ros::Publisher pubPath =
      nh.advertise<visualization_msgs::MarkerArray>("descriptor_line", 10);
  ros::Publisher pubCurrentPose =
      nh.advertise<nav_msgs::Odometry>("/current_pose", 10);
  ros::Publisher pubMatchedPose =
      nh.advertise<nav_msgs::Odometry>("/matched_pose", 10);
  ros::Publisher pubMatchedCloud =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_matched", 100);
  ros::Publisher pubMatchedBinary =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_matched_key_points", 100);
  ros::Publisher pubLoopStatus =
      nh.advertise<visualization_msgs::MarkerArray>("/loop_status", 100);
  ros::Publisher pubBTC =
      nh.advertise<visualization_msgs::MarkerArray>("descriptor_line", 10);

  std_msgs::ColorRGBA color_tp;
  std_msgs::ColorRGBA color_fp;
  std_msgs::ColorRGBA color_path;
  double scale_tp = 4.0;
  double scale_fp = 5.0;
  double scale_path = 3.0;
  color_tp.a = 1.0;
  color_tp.r = 0.0 / 255.0;
  color_tp.g = 255.0 / 255.0;
  color_tp.b = 0.0 / 255.0;

  color_fp.a = 1.0;
  color_fp.r = 1.0;
  color_fp.g = 0.0;
  color_fp.b = 0.0;

  color_path.a = 0.8;
  color_path.r = 255.0 / 255.0;
  color_path.g = 255.0 / 255.0;
  color_path.b = 255.0 / 255.0;

  ros::Rate loop(50000);
  ros::Rate slow_loop(1000);

  ConfigSetting config_setting;
  load_config_setting(setting_path, config_setting);

  // pose is only for visulization and gt overlap calculation
  std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> pose_list;
  std::vector<double> time_list;
  load_evo_pose_with_time(pose_file, pose_list, time_list);
  std::string print_msg = "Successfully load pose file:" + pose_file +
                          ". pose size:" + std::to_string(time_list.size());
  ROS_INFO_STREAM(print_msg.c_str());

  BtcDescManager *btc_manager = new BtcDescManager(config_setting);
  btc_manager->print_debug_info_ = false;
  pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud(
      new pcl::PointCloud<pcl::PointXYZI>());

  std::vector<double> descriptor_time;
  std::vector<double> querying_time;
  std::vector<double> update_time;
  int triggle_loop_num = 0;
  int true_loop_num = 0;
  bool finish = false;

  pcl::PCDReader reader;

  while (ros::ok() && !finish) {
    for (size_t submap_id = 0; submap_id < pose_list.size(); ++submap_id) {
      pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(
          new pcl::PointCloud<pcl::PointXYZI>());
      pcl::PointCloud<pcl::PointXYZI> transform_cloud;
      cloud->reserve(1000000);
      transform_cloud.reserve(1000000);
      // Get pose information, only for gt overlap calculation
      Eigen::Vector3d translation = pose_list[submap_id].first;
      Eigen::Matrix3d rotation = pose_list[submap_id].second;
      if (read_bin) {
        std::stringstream ss;
        ss << pcds_dir << "/" << std::setfill('0') << std::setw(6) << submap_id
           << ".bin";
        std::string pcd_file = ss.str();
        auto t_load_start = std::chrono::high_resolution_clock::now();
        std::vector<float> lidar_data = read_lidar_data(ss.str());
        if (lidar_data.size() == 0) {
          break;
        }

        for (std::size_t i = 0; i < lidar_data.size(); i += 4) {
          pcl::PointXYZI point;
          point.x = lidar_data[i];
          point.y = lidar_data[i + 1];
          point.z = lidar_data[i + 2];
          point.intensity = lidar_data[i + 3];
          Eigen::Vector3d pv = point2vec(point);
          pv = rotation * pv + translation;
          cloud->points.push_back(point);
          point = vec2point(pv);
          transform_cloud.points.push_back(vec2point(pv));
        }
        auto t_load_end = std::chrono::high_resolution_clock::now();
        std::cout << "[Time] load cloud: " << time_inc(t_load_end, t_load_start)
                  << "ms, " << std::endl;
      } else {
        // Load point cloud from pcd file
        std::stringstream ss;
        ss << pcds_dir << "/" << std::setfill('0') << std::setw(6) << submap_id
           << ".pcd";
        std::string pcd_file = ss.str();

        auto t_load_start = std::chrono::high_resolution_clock::now();
        // pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_file, *cloud)
        if (reader.read(pcd_file, *cloud) == -1) {
          ROS_ERROR_STREAM("Couldn't read file " << pcd_file);
          continue;
        }
        auto t_load_end = std::chrono::high_resolution_clock::now();
        std::cout << "[Time] load cloud: " << time_inc(t_load_end, t_load_start)
                  << "ms, " << std::endl;
        transform_cloud = *cloud;

        for (size_t j = 0; j < transform_cloud.size(); j++) {
          Eigen::Vector3d pv = point2vec(transform_cloud.points[j]);
          pv = rotation * pv + translation;
          transform_cloud.points[j] = vec2point(pv);
        }
      }

      // step1. Descriptor Extraction
      std::cout << "[Description] submap id:" << submap_id << std::endl;
      auto t_descriptor_begin = std::chrono::high_resolution_clock::now();
      std::vector<BTC> btcs_vec;
      btc_manager->GenerateBtcDescs(transform_cloud.makeShared(), submap_id,
                                    btcs_vec);
      auto t_descriptor_end = std::chrono::high_resolution_clock::now();
      descriptor_time.push_back(time_inc(t_descriptor_end, t_descriptor_begin));
      // step2. Searching Loop
      auto t_query_begin = std::chrono::high_resolution_clock::now();
      std::pair<int, double> search_result(-1, 0);
      std::pair<Eigen::Vector3d, Eigen::Matrix3d> loop_transform;
      loop_transform.first << 0, 0, 0;
      loop_transform.second = Eigen::Matrix3d::Identity();
      std::vector<std::pair<BTC, BTC>> loop_std_pair;
      if (submap_id > config_setting.skip_near_num_) {
        if (btcs_vec.size() == 0) {
          // getchar();
        }
        btc_manager->SearchLoop(btcs_vec, search_result, loop_transform,
                                loop_std_pair);
      }
      if (search_result.first > 0) {
        std::cout << "[Loop Detection] triggle loop: " << submap_id << "--"
                  << search_result.first << ", score:" << search_result.second
                  << std::endl;
      }
      auto t_query_end = std::chrono::high_resolution_clock::now();
      querying_time.push_back(time_inc(t_query_end, t_query_begin));

      // step3. Add descriptors to the database
      auto t_map_update_begin = std::chrono::high_resolution_clock::now();
      btc_manager->AddBtcDescs(btcs_vec);
      auto t_map_update_end = std::chrono::high_resolution_clock::now();
      update_time.push_back(time_inc(t_map_update_end, t_map_update_begin));
      std::cout << "[Time] descriptor extraction: "
                << time_inc(t_descriptor_end, t_descriptor_begin) << "ms, "
                << "query: " << time_inc(t_query_end, t_query_begin) << "ms, "
                << "update map:"
                << time_inc(t_map_update_end, t_map_update_begin) << "ms"
                << std::endl;
      std::cout << std::endl;

      // down sample to save memory
      down_sampling_voxel(transform_cloud, 0.5);
      btc_manager->key_cloud_vec_.push_back(transform_cloud.makeShared());

      // visuliazaion
      sensor_msgs::PointCloud2 pub_cloud;
      pcl::toROSMsg(transform_cloud, pub_cloud);
      pub_cloud.header.frame_id = "camera_init";
      pubCureentCloud.publish(pub_cloud);

      pcl::PointCloud<pcl::PointXYZ> key_points_cloud;
      for (auto var : btc_manager->history_binary_list_.back()) {
        pcl::PointXYZ pi;
        pi.x = var.location_[0];
        pi.y = var.location_[1];
        pi.z = var.location_[2];
        key_points_cloud.push_back(pi);
      }
      pcl::toROSMsg(key_points_cloud, pub_cloud);
      pub_cloud.header.frame_id = "camera_init";
      pubCurrentBinary.publish(pub_cloud);

      visualization_msgs::MarkerArray marker_array;
      visualization_msgs::Marker marker;
      marker.header.frame_id = "camera_init";
      marker.ns = "colored_path";
      marker.id = submap_id;
      marker.type = visualization_msgs::Marker::LINE_LIST;
      marker.action = visualization_msgs::Marker::ADD;
      marker.pose.orientation.w = 1.0;
      if (search_result.first >= 0) {
        triggle_loop_num++;
        Eigen::Matrix4d transform1 = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d transform2 = Eigen::Matrix4d::Identity();
        publish_std(loop_std_pair, transform1, transform2, pubBTC);
        slow_loop.sleep();
        double cloud_overlap =
            calc_overlap(transform_cloud.makeShared(),
                         btc_manager->key_cloud_vec_[search_result.first], 0.5);
        pcl::PointCloud<pcl::PointXYZ> match_key_points_cloud;
        for (auto var :
             btc_manager->history_binary_list_[search_result.first]) {
          pcl::PointXYZ pi;
          pi.x = var.location_[0];
          pi.y = var.location_[1];
          pi.z = var.location_[2];
          match_key_points_cloud.push_back(pi);
        }
        pcl::toROSMsg(match_key_points_cloud, pub_cloud);
        pub_cloud.header.frame_id = "camera_init";
        pubMatchedBinary.publish(pub_cloud);
        // true positive
        if (cloud_overlap >= cloud_overlap_thr) {
          true_loop_num++;
          pcl::PointCloud<pcl::PointXYZRGB> matched_cloud;
          matched_cloud.resize(
              btc_manager->key_cloud_vec_[search_result.first]->size());
          for (size_t i = 0;
               i < btc_manager->key_cloud_vec_[search_result.first]->size();
               i++) {
            pcl::PointXYZRGB pi;
            pi.x =
                btc_manager->key_cloud_vec_[search_result.first]->points[i].x;
            pi.y =
                btc_manager->key_cloud_vec_[search_result.first]->points[i].y;
            pi.z =
                btc_manager->key_cloud_vec_[search_result.first]->points[i].z;
            pi.r = 0;
            pi.g = 255;
            pi.b = 0;
            matched_cloud.points[i] = pi;
          }
          pcl::toROSMsg(matched_cloud, pub_cloud);
          pub_cloud.header.frame_id = "camera_init";
          pubMatchedCloud.publish(pub_cloud);
          slow_loop.sleep();

          marker.scale.x = scale_tp;
          marker.color = color_tp;
          geometry_msgs::Point point1;
          point1.x = pose_list[submap_id - 1].first[0];
          point1.y = pose_list[submap_id - 1].first[1];
          point1.z = pose_list[submap_id - 1].first[2];
          geometry_msgs::Point point2;
          point2.x = pose_list[submap_id].first[0];
          point2.y = pose_list[submap_id].first[1];
          point2.z = pose_list[submap_id].first[2];
          marker.points.push_back(point1);
          marker.points.push_back(point2);

        } else {
          pcl::PointCloud<pcl::PointXYZRGB> matched_cloud;
          matched_cloud.resize(
              btc_manager->key_cloud_vec_[search_result.first]->size());
          for (size_t i = 0;
               i < btc_manager->key_cloud_vec_[search_result.first]->size();
               i++) {
            pcl::PointXYZRGB pi;
            pi.x =
                btc_manager->key_cloud_vec_[search_result.first]->points[i].x;
            pi.y =
                btc_manager->key_cloud_vec_[search_result.first]->points[i].y;
            pi.z =
                btc_manager->key_cloud_vec_[search_result.first]->points[i].z;
            pi.r = 255;
            pi.g = 0;
            pi.b = 0;
            matched_cloud.points[i] = pi;
          }
          pcl::toROSMsg(matched_cloud, pub_cloud);
          pub_cloud.header.frame_id = "camera_init";
          pubMatchedCloud.publish(pub_cloud);
          slow_loop.sleep();
          marker.scale.x = scale_fp;
          marker.color = color_fp;
          geometry_msgs::Point point1;
          point1.x = pose_list[submap_id - 1].first[0];
          point1.y = pose_list[submap_id - 1].first[1];
          point1.z = pose_list[submap_id - 1].first[2];
          geometry_msgs::Point point2;
          point2.x = pose_list[submap_id].first[0];
          point2.y = pose_list[submap_id].first[1];
          point2.z = pose_list[submap_id].first[2];
          marker.points.push_back(point1);
          marker.points.push_back(point2);
        }

      } else {
        if (submap_id > 0) {
          marker.scale.x = scale_path;
          marker.color = color_path;
          geometry_msgs::Point point1;
          point1.x = pose_list[submap_id - 1].first[0];
          point1.y = pose_list[submap_id - 1].first[1];
          point1.z = pose_list[submap_id - 1].first[2];
          geometry_msgs::Point point2;
          point2.x = pose_list[submap_id].first[0];
          point2.y = pose_list[submap_id].first[1];
          point2.z = pose_list[submap_id].first[2];
          marker.points.push_back(point1);
          marker.points.push_back(point2);
        }
      }
      marker_array.markers.push_back(marker);
      pubLoopStatus.publish(marker_array);
      loop.sleep();
    }
    finish = true;
  }

  double mean_descriptor_time =
      std::accumulate(descriptor_time.begin(), descriptor_time.end(), 0) * 1.0 /
      descriptor_time.size();
  double mean_query_time =
      std::accumulate(querying_time.begin(), querying_time.end(), 0) * 1.0 /
      querying_time.size();
  double mean_update_time =
      std::accumulate(update_time.begin(), update_time.end(), 0) * 1.0 /
      update_time.size();
  std::cout << "Total submap number:" << pose_list.size()
            << ", triggle loop number:" << triggle_loop_num
            << ", true loop number:" << true_loop_num << std::endl;
  std::cout << "Mean time for descriptor extraction: " << mean_descriptor_time
            << "ms, query: " << mean_query_time
            << "ms, update: " << mean_update_time << "ms, total: "
            << mean_descriptor_time + mean_query_time + mean_update_time << "ms"
            << std::endl;
  return 0;
}