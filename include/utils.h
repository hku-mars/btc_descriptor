
#ifndef UTILS_H
#define UTILS_H
#include <pcl/common/io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <fstream>
#include <string>

void load_pose_with_time(
    const std::string &pose_file,
    std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> &pose_list,
    std::vector<double> &time_list);

void load_evo_pose_with_time(
    const std::string &pose_file,
    std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> &pose_list,
    std::vector<double> &time_list);

void load_cu_pose_with_time(
    const std::string &pose_file,
    std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> &pose_list,
    std::vector<double> &time_list);

void load_pose_with_frame(
    const std::string &pose_file,
    std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> &pose_list,
    std::vector<int> &frame_number_list);

int findPoseIndexUsingTime(std::vector<double> &time_list, double &time);

pcl::PointXYZI vec2point(const Eigen::Vector3d &vec);
// Eigen::Vector3d point2vec(const pcl::PointXYZI &pi);

Eigen::Vector3d normal2vec(const pcl::PointXYZINormal &pi);

template <typename T>
Eigen::Vector3d point2vec(const T &pi) {
  Eigen::Vector3d vec(pi.x, pi.y, pi.z);
  return vec;
}

double time_inc(std::chrono::_V2::system_clock::time_point &t_end,
                std::chrono::_V2::system_clock::time_point &t_begin);

double calc_overlap(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud1,
                    const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud2,
                    double dis_threshold);

double calc_overlap(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud1,
                    const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud2,
                    double dis_threshold, int skip_num);

#endif