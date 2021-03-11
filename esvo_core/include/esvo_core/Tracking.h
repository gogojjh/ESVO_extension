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
    void timeSurfaceCallback(const sensor_msgs::ImageConstPtr &time_surface_left);
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
    ros::Subscriber map_sub_;
    ros::Subscriber TS_left_sub_;
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

    std::vector<std::unordered_map<std::string, double> > vTimeCost_;

    Eigen::Quaterniond q_gt_s_;
    Eigen::Vector3d t_gt_s_;

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
    std::deque<std::pair<ros::Time, TimeSurfaceObservation>> TS_buf_;

    /**** offline parameters ***/
    size_t tracking_rate_hz_;
    size_t TS_HISTORY_LENGTH_;
    size_t REF_HISTORY_LENGTH_;
    bool bSaveTrajectory_;
    bool bVisualizeTrajectory_;
    std::string resultPath_;
    std::string strDataset_, strSequence_, strRep_;

    Eigen::Matrix<double, 4, 4> T_world_cur_;
    Eigen::Matrix<double, 4, 4> T_world_map_;

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
