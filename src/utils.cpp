#include "include/utils.h"
void load_pose_with_time(
    const std::string &pose_file,
    std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> &pose_list,
    std::vector<double> &time_list) {
  time_list.clear();
  pose_list.clear();
  std::ifstream fin(pose_file);
  std::string line;
  Eigen::Matrix<double, 1, 7> temp_matrix;
  while (getline(fin, line)) {
    std::istringstream sin(line);
    std::vector<std::string> Waypoints;
    std::string info;
    int number = 0;
    while (getline(sin, info, ',')) {
      if (number == 0) {
        double time;
        std::stringstream data;
        data << info;
        data >> time;
        time_list.push_back(time);
        number++;
      } else {
        double p;
        std::stringstream data;
        data << info;
        data >> p;
        temp_matrix[number - 1] = p;
        if (number == 7) {
          Eigen::Vector3d translation(temp_matrix[0], temp_matrix[1],
                                      temp_matrix[2]);
          Eigen::Quaterniond q(temp_matrix[3], temp_matrix[4], temp_matrix[5],
                               temp_matrix[6]);
          std::pair<Eigen::Vector3d, Eigen::Matrix3d> single_pose;
          single_pose.first = translation;
          single_pose.second = q.toRotationMatrix();
          pose_list.push_back(single_pose);
        }
        number++;
      }
    }
  }
}

void load_cu_pose_with_time(
    const std::string &pose_file,
    std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> &pose_list,
    std::vector<double> &time_list) {
  time_list.clear();
  pose_list.clear();
  std::ifstream fin(pose_file);
  std::string line;
  Eigen::Matrix<double, 1, 12> temp_matrix;
  while (getline(fin, line)) {
    std::istringstream sin(line);
    std::vector<std::string> Waypoints;
    std::string info;
    int number = 0;
    while (getline(sin, info, ',')) {
      if (number == 0) {
        double time;
        std::stringstream data;
        data << info;
        data >> time;
        time_list.push_back(time);
        number++;
      } else {
        double p;
        std::stringstream data;
        data << info;
        data >> p;
        temp_matrix[number - 1] = p;
        if (number == 12) {
          Eigen::Vector3d translation(temp_matrix[3], temp_matrix[7],
                                      temp_matrix[11]);
          Eigen::Matrix3d rotation;
          rotation << temp_matrix[0], temp_matrix[1], temp_matrix[2],
              temp_matrix[4], temp_matrix[5], temp_matrix[6], temp_matrix[8],
              temp_matrix[9], temp_matrix[10];
          std::pair<Eigen::Vector3d, Eigen::Matrix3d> single_pose;
          single_pose.first = translation;
          single_pose.second = rotation;
          pose_list.push_back(single_pose);
        }
        number++;
      }
    }
  }
}

void load_pose_with_frame(
    const std::string &pose_file,
    std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> &pose_list,
    std::vector<int> &frame_number_list) {
  frame_number_list.clear();
  pose_list.clear();
  std::ifstream fin(pose_file);
  std::string line;
  Eigen::Matrix<double, 1, 12> temp_matrix;
  while (getline(fin, line)) {
    std::istringstream sin(line);
    std::vector<std::string> Waypoints;
    std::string info;
    int number = 0;
    while (getline(sin, info, ' ')) {
      if (number == 0) {
        int frame_number;
        std::stringstream data;
        data << info;
        data >> frame_number;
        frame_number_list.push_back(frame_number);
        number++;
      } else {
        double p;
        std::stringstream data;
        data << info;
        data >> p;
        temp_matrix[number - 1] = p;
        if (number == 12) {
          Eigen::Vector3d translation(temp_matrix[3], temp_matrix[7],
                                      temp_matrix[11]);
          Eigen::Matrix3d rotation;
          rotation << temp_matrix[0], temp_matrix[1], temp_matrix[2],
              temp_matrix[4], temp_matrix[5], temp_matrix[6], temp_matrix[8],
              temp_matrix[9], temp_matrix[10];
          std::pair<Eigen::Vector3d, Eigen::Matrix3d> single_pose;
          single_pose.first = translation;
          single_pose.second = rotation;
          pose_list.push_back(single_pose);
        }
        number++;
      }
    }
  }
}

void load_evo_pose_with_time(
    const std::string &pose_file,
    std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> &pose_list,
    std::vector<double> &time_list) {
  time_list.clear();
  pose_list.clear();
  std::ifstream fin(pose_file);
  std::string line;
  Eigen::Matrix<double, 1, 7> temp_matrix;
  while (getline(fin, line)) {
    std::istringstream sin(line);
    std::vector<std::string> Waypoints;
    std::string info;
    int number = 0;
    while (getline(sin, info, ' ')) {
      if (number == 0) {
        double time;
        std::stringstream data;
        data << info;
        data >> time;
        time_list.push_back(time);
        number++;
      } else {
        double p;
        std::stringstream data;
        data << info;
        data >> p;
        temp_matrix[number - 1] = p;
        if (number == 7) {
          Eigen::Vector3d translation(temp_matrix[0], temp_matrix[1],
                                      temp_matrix[2]);
          Eigen::Quaterniond q(temp_matrix[6], temp_matrix[3], temp_matrix[4],
                               temp_matrix[5]);
          std::pair<Eigen::Vector3d, Eigen::Matrix3d> single_pose;
          single_pose.first = translation;
          single_pose.second = q.toRotationMatrix();
          pose_list.push_back(single_pose);
        }
        number++;
      }
    }
  }
}

int findPoseIndexUsingTime(std::vector<double> &time_list, double &time) {
  double time_inc = 10000000000;
  int min_index = -1;
  for (size_t i = 0; i < time_list.size(); i++) {
    if (fabs(time_list[i] - time) < time_inc) {
      time_inc = fabs(time_list[i] - time);
      min_index = i;
    }
  }
  return min_index;
}

pcl::PointXYZI vec2point(const Eigen::Vector3d &vec) {
  pcl::PointXYZI pi;
  pi.x = vec[0];
  pi.y = vec[1];
  pi.z = vec[2];
  return pi;
}
// Eigen::Vector3d point2vec(const pcl::PointXYZI &pi) {
//   return Eigen::Vector3d(pi.x, pi.y, pi.z);
// }

Eigen::Vector3d normal2vec(const pcl::PointXYZINormal &pi) {
  Eigen::Vector3d vec(pi.normal_x, pi.normal_y, pi.normal_z);
  return vec;
}

double time_inc(std::chrono::_V2::system_clock::time_point &t_end,
                std::chrono::_V2::system_clock::time_point &t_begin) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(t_end -
                                                                   t_begin)
             .count() *
         1000;
}

double calc_overlap(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud1,
                    const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud2,
                    double dis_threshold) {
  int point_kip = 2;
  double match_num = 0;
  pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZI>);
  kd_tree->setInputCloud(cloud2);
  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);
  for (size_t i = 0; i < cloud1->size(); i += point_kip) {
    pcl::PointXYZI searchPoint = cloud1->points[i];
    if (kd_tree->nearestKSearch(searchPoint, 1, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0) {
      if (pointNKNSquaredDistance[0] < dis_threshold * dis_threshold) {
        match_num++;
      }
    }
  }
  // std::cout << "cloud1 size:" << cloud1->size()
  //           << " cloud2 size: " << cloud2->size() << " match size:" <<
  //           match_num
  //           << std::endl;
  double overlap =
      2 * match_num * point_kip / (cloud1->size() + cloud2->size());
  return overlap;
}

double calc_overlap(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud1,
                    const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud2,
                    double dis_threshold, int skip_num) {
  double match_num = 0;
  pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZI>);
  kd_tree->setInputCloud(cloud2);
  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);
  for (size_t i = 0; i < cloud1->size(); i += skip_num) {
    pcl::PointXYZI searchPoint = cloud1->points[i];
    if (kd_tree->nearestKSearch(searchPoint, 1, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0) {
      if (pointNKNSquaredDistance[0] < dis_threshold * dis_threshold) {
        match_num++;
      }
    }
  }
  // std::cout << "cloud1 size:" << cloud1->size()
  //           << " cloud2 size: " << cloud2->size() << " match size:" <<
  //           match_num
  //           << std::endl;
  double overlap =
      (2 * match_num * skip_num) / (cloud1->size() + cloud2->size());
  return overlap;
}
