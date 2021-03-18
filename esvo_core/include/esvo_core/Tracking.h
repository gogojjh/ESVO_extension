#ifndef ESVO_CORE_MONOTRACKING_H
#define ESVO_CORE_MONOTRACKING_H

#include <nav_msgs/Path.h>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <tf2_ros/transform_broadcaster.h>
#include <std_msgs/Int16.h>

#include <esvo_core/container/CameraSystem.h>
#include <esvo_core/core/RegProblemLM.h>
#include <esvo_core/core/RegProblemSolverLM.h>
#include <esvo_core/tools/utils.h>
#include <esvo_core/tools/cayley.h>
#include <esvo_core/tools/Visualization.h>
#include <initial/InitialMotionEstimator.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <cv_bridge/cv_bridge.h>

#include <map>
#include <deque>
#include <mutex>
#include <future>

#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>

// #define ESVO_CORE_MONO_TRACKING_DEBUG
// #define ESVO_CORE_TRACKING_LOG

namespace esvo_core
{
  using namespace core;
  enum TrackingStatus
  {
    IDLE,
    WORKING
  };

  // class PosePredictor
  // {
  // public:
  //   PosePredictor()
  //   {
  //     T0_.setIdentity();
  //     T1_.setIdentity();
  //     T_pred_.setIdentity();
  //     t0_ = ros::Time::now();
  //     t1_ = ros::Time::now();
  //   }

  //   void reset()
  //   {
  //     T0_.setIdentity();
  //     T1_.setIdentity();
  //     T_pred_.setIdentity();
  //     t0_ = ros::Time::now();
  //     t1_ = ros::Time::now();
  //   }

  //   void predict(const ros::Time &t)
  //   {
  //     // Linear interpolation in SE(3)
  //     const Eigen::Matrix4d T_relative = T0_.inverse() * T1_;
  //     const double delta_t = (t - t0_).toSec() / (t1_ - t0_).toSec();
  //     T_pred_ = T0_ * getLinearInterpolation(T_relative, delta_t);
  //   }

  //   Eigen::Matrix4d getLinearInterpolation(const Eigen::Matrix4d &T_relative, const double &delta_t) const
  //   {
  //     Eigen::Matrix3d R_relative = T_relative.topLeftCorner<3, 3>();
  //     Eigen::Quaterniond q_relative(R_relative);
  //     Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  //     T.topLeftCorner<3, 3>() = q_relative.slerp(delta_t, q_relative).toRotationMatrix();
  //     T.topRightCorner<3, 1>() = delta_t * T_relative.topRightCorner<3, 1>();
  //     return T;
  //   }

  //   Eigen::Matrix4d T0_, T1_, T_pred_;
  //   ros::Time t0_, t1_;
  // };

  class Tracking
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Tracking(const ros::NodeHandle &nh, const ros::NodeHandle &nh_private);
    virtual ~Tracking();

    // functions regarding tracking
    void TrackingLoopTS();
    void TrackingLoopEM();
    void TrackingLoopTSEM();
    bool refDataTransferring();
    bool curDataTransferring(); // These two data transferring functions are decoupled because the data are not updated at the same frequency.

    // topic callback functions
    void refMapCallback(const sensor_msgs::PointCloud2::ConstPtr &msg);
    void timeSurfaceCallback(
        const sensor_msgs::ImageConstPtr &time_surface_left,
        const sensor_msgs::ImageConstPtr &time_surface_right);
    void eventsCallback(const dvs_msgs::EventArray::ConstPtr &msg);

    // results
    void publishPose(const ros::Time &t, Transformation &tr);
    void publishPath(const ros::Time &t, Transformation &tr);
    // void saveTrajectory(const std::string &resultDir);
    void saveTrajectory(const std::string &resultDir,
                        const std::list<std::string> &lTimestamp_,
                        const std::list<Eigen::Matrix<double, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 4>>> &lPose_);
    void saveTimeCost(const std::string &resultDir,
                      const std::vector<std::unordered_map<std::string, double>> &vTimeCost_);

    // utils
    void reset();
    void clearEventQueue();
    void stampedPoseCallback(const geometry_msgs::PoseStampedConstPtr &msg);
    void gtPoseCallback(const geometry_msgs::PoseStampedConstPtr &msg);
    bool getPoseAt(const ros::Time &t,
                   esvo_core::Transformation &Tr, // T_world_something
                   const std::string &source_frame);

  private:
    ros::NodeHandle nh_, pnh_;
    image_transport::ImageTransport it_;

    // subscribers and publishers
    ros::Subscriber events_left_sub_;
    message_filters::Subscriber<sensor_msgs::Image> TS_left_sub_, TS_right_sub_;

    ros::Subscriber map_sub_;
    ros::Subscriber gtPose_sub_, stampedPose_sub_;
    image_transport::Publisher reprojMap_pub_left_;

    // publishers
    ros::Publisher pose_pub_, path_pub_, pose_gt_pub_, path_gt_pub_;

    // results
    nav_msgs::Path path_, path_gt_;
    std::list<Eigen::Matrix<double, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 4>>> lPose_;
    std::list<std::string> lTimestamp_;
    std::list<Eigen::Matrix<double, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 4>>> lPose_GT_;
    std::list<std::string> lTimestamp_GT_;
    double last_gt_timestamp_, last_save_trajectory_timestamp_;

    std::vector<std::unordered_map<std::string, double>> vTimeCost_;
    std::vector<std::unordered_map<std::string, double>> vLambda_;

    Eigen::Quaterniond q_gt_s_;
    Eigen::Vector3d t_gt_s_;

    // Time Surface sync policy
    typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image> ExactSyncPolicy;
    message_filters::Synchronizer<ExactSyncPolicy> TS_sync_;

    // offline data
    std::string dvs_frame_id_;
    std::string world_frame_id_;
    std::string calibInfoDir_;
    CameraSystem::Ptr camSysPtr_;

    // inter-thread management
    std::mutex data_mutex_;
    std::mutex m_buf_;

    // online data
    EventQueue events_left_;
    size_t TS_id_;
    std::shared_ptr<tf::Transformer> tf_;
    RefFrame ref_;
    CurFrame cur_;

    std::deque<std::pair<ros::Time, PointCloud::Ptr>> refPCMap_buf_;
    std::deque<std::pair<ros::Time, TimeSurfaceObservation>> TS_history_;

    /**** offline parameters ***/
    size_t tracking_rate_hz_;
    size_t TS_HISTORY_LENGTH_;
    size_t REF_HISTORY_LENGTH_;
    bool bSaveTrajectory_;
    bool bVisualizeTrajectory_;
    std::string resultPath_;
    std::string strDataset_, strSequence_, strRep_;
    size_t eventNum_EM_;

    Eigen::Matrix<double, 4, 4> T_world_cur_;
    Eigen::Matrix<double, 4, 4> T_world_map_;
    // PosePredictor pose_predictor_;

    /*** system objects ***/
    RegProblemType rpType_;
    TrackingStatus ets_;
    std::string ESVO_System_Status_;
    RegProblemConfig::Ptr rpConfigPtr_;
    RegProblemSolverLM rpSolver_;

    size_t num_NewEvents_;
  };
} // namespace esvo_core

#endif //ESVO_CORE_TRACKING_H
