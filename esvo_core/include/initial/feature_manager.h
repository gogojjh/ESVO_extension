/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <glog/logging.h>

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

class FeaturePerFrame
{
public:
  FeaturePerFrame(const Eigen::Matrix<double, 5, 1> &_point)
  {
    point.x() = _point(0);
    point.y() = _point(1);
    point.z() = _point(2);
    uv.x() = _point(3);
    uv.y() = _point(4);
  }
  Eigen::Vector3d point;
  Eigen::Vector2d uv;
};

class FeaturePerId
{
public:
  const int feature_id;
  int start_frame;
  std::vector<FeaturePerFrame> feature_per_frame; // the position of feature_id^th feature
  int used_num;
  double estimated_depth;
  int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

  FeaturePerId(int _feature_id, int _start_frame)
      : feature_id(_feature_id), start_frame(_start_frame),
        used_num(0), estimated_depth(-1.0), solve_flag(0)
  {
  }

  int endFrame();
};

class FeatureManager
{
public:
  FeatureManager();
  ~FeatureManager() = default;

  void clearState();
  int getFeatureCount();
  // add feature and select keyframes
  bool addFeatureCheckParallax(int frame_count, const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);
  // set feature depth after the optimization
  void setDepth(const Eigen::VectorXd &x);
  // remove outlier features whose depth is not optimized
  void removeFailures();
  // set all features depth as -1 (initialze)
  void clearDepth();
  Eigen::VectorXd getDepthVector();
  // triangulate points and initialize their coordinates into the local frame;
  void triangulate(Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d ric[]);
  void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
  void removeBack();
  void removeFront(int frame_count);
  void removeOutlier(std::set<int> &outlierIndex);

  // ----------------------------------------------------------------
  // modified by jjiao
  // void triangulate(const CircularBuffer<Eigen::Vector3d> &Ps, const CircularBuffer<Eigen::Quaterniond> &Qs,
  //                  const Eigen::Vector3d &tbc, const Eigen::Quaterniond &qbc);
  void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0,
                        Eigen::Matrix<double, 3, 4> &Pose1,
                        Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d);

  // implement PnP method
  bool solvePoseByPnP(Eigen::Matrix3d &R, Eigen::Vector3d &P, std::vector<cv::Point2f> &pts2D, std::vector<cv::Point3f> &pts3D);
  void initFramePoseByPnP(int frameCnt, Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d ric[]);
  void initDepth();
  void removeBackShiftDepth(const Eigen::Quaterniond &marg_Q, const Eigen::Vector3d &marg_P, const Eigen::Quaterniond &new_Q, const Eigen::Vector3d &new_P);

  std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> getCorresponding2D(const int &frame_count_l, const int &frame_count_r);
  void printFeatureInfo() const;
  void savePNPData(const int &frameCnt, const CircularBuffer<Eigen::Vector3d> &Ps, const CircularBuffer<Eigen::Quaterniond> &Qs,
                  const Eigen::Vector3d &tbc, const Eigen::Quaterniond &qbc, const cv::Mat &img);

  // ----------------------------------------------------------------
  std::list<FeaturePerId> feature_;

  int last_track_num_;
  double last_average_parallax_;
  int new_feature_num_;
  int long_track_num_;

private:
  // calculate the parallax of each feature
  double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
};

#endif
