#ifndef TRAJECTORY_HPP_
#define TRAJECTORY_HPP_

#include <glog/logging.h>

#include <map>
#include <iostream>
#include <ros/time.h>
#include <eigen3/Eigen/Dense>

typedef std::map<ros::Time, Eigen::Matrix4d> PoseMap;

template <class DerivedTrajectory>
class Trajectory
{
public:

  DerivedTrajectory &derived()
  {
    return static_cast<DerivedTrajectory &>(*this);
  }

  Trajectory() {}

  Trajectory(const PoseMap &poses)
      : poses_(poses)
  {
  }

  // Returns T_W_C (mapping points from C to the world frame W)
  bool getPoseAt(const ros::Time &t, Eigen::Matrix4d &T) const
  {
    return derived().getPoseAt(t, T);
  }

  void getFirstControlPose(Eigen::Matrix4d *T, ros::Time *t) const
  {
    *t = poses_.begin()->first;
    *T = poses_.begin()->second;
  }

  void getLastControlPose(Eigen::Matrix4d *T, ros::Time *t) const
  {
    *t = poses_.rbegin()->first;
    *T = poses_.rbegin()->second;
  }

  size_t getNumControlPoses() const
  {
    return poses_.size();
  }

  bool print() const
  {
    size_t control_pose_idx = 0u;
    for (auto it : poses_)
    {
      std::cout << "--Control pose #" << control_pose_idx++ << ". time = " << it.first << std::endl;
      std::cout << "--T = " << std::endl;
      std::cout << it.second << std::endl;
    }
    return true;
  }

protected:
  PoseMap poses_;
};

class LinearTrajectory : public Trajectory<LinearTrajectory>
{
public:
  LinearTrajectory() : Trajectory() {}

  LinearTrajectory(const PoseMap &poses)
      : Trajectory(poses)
  {
    CHECK_GE(poses_.size(), 2u) << "At least two poses need to be provided";
  }

  bool getPoseAt(const ros::Time &t, Eigen::Matrix4d &T) const
  {
    ros::Time t0_, t1_;
    Eigen::Matrix4d T0_, T1_;

    // Check if it is between two known poses
    auto it1 = poses_.upper_bound(t);
    if (it1 == poses_.begin())
    {
      std::cout << "Cannot extrapolate in the past. Requested pose: "
                << t << " but the earliest pose available is at time: "
                << poses_.begin()->first;
      return false;
    }
    else if (it1 == poses_.end())
    {
      std::cout << "Cannot extrapolate in the future. Requested pose: "
                << t << " but the latest pose available is at time: "
                << (poses_.rbegin())->first;
      return false;
    }
    else
    {
      auto it0 = std::prev(it1);
      t0_ = (it0)->first;
      T0_ = (it0)->second;
      t1_ = (it1)->first;
      T1_ = (it1)->second;
    }

    // Linear interpolation in SE(3)
    const Eigen::Matrix4d T_relative = T0_.inverse() * T1_;
    const double delta_t = (t - t0_).toSec() / (t1_ - t0_).toSec();
    T = T0_ * getLinearInterpolation(T_relative, delta_t);
    return true;
  }

  Eigen::Matrix4d getLinearInterpolation(const Eigen::Matrix4d &T_relative, const double &delta_t) const
  {
    Eigen::Matrix3d R_relative = T_relative.topLeftCorner<3, 3>();
    Eigen::Quaterniond q_relative(R_relative);
    Eigen::Quaterniond q = q_relative.slerp(delta_t, q_relative);
    Eigen::Vector3d t = delta_t * T_relative.topRightCorner<3, 1>();

    Eigen::Matrix4d T;
    T.setIdentity();
    T.topLeftCorner<3, 3>() = q.toRotationMatrix();
    T.topRightCorner<3, 1>() = t;
    return T;
  }
};

#endif