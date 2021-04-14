#ifndef LIDAR_DEPTH_MAP_H_
#define LIDAR_DEPTH_MAP_H_

#include <nav_msgs/Path.h>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>

#include <tf2_ros/transform_broadcaster.h>
#include <std_msgs/Int16.h>

#include <esvo_core/container/CameraSystem.h>
#include <esvo_core/container/SensorSystem.h>
#include <esvo_core/tools/utils.h>
#include <esvo_core/tools/cayley.h>
#include <esvo_core/tools/Trajectory.hpp>
#include <esvo_core/tools/Visualization.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <cv_bridge/cv_bridge.h>

#include <map>
#include <vector>
#include <deque>
#include <mutex>
#include <future>

#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl_ros/point_cloud.h>

namespace esvo_core
{
    class LiDARDepthMap
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        LiDARDepthMap(const ros::NodeHandle &nh, const ros::NodeHandle &nh_private);
        virtual ~LiDARDepthMap();

        // functions regarding LiDARDepthMap
        void DepthMapLoop();

        // topic callback functions
        void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg);
        void stampedPoseCallback(const geometry_msgs::PoseStampedConstPtr &msg);
        void gtPoseCallback(const geometry_msgs::PoseStampedConstPtr &msg);

        // publish results
        void publishDepthMap(const ros::Time &t);
        void saveDepthMap(const std::string &resultDir, const pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_ptr);
        void reset();

    private:
        ros::NodeHandle nh_, pnh_;
        image_transport::ImageTransport it_;

        // subscribers and publishers
        ros::Subscriber cloud_sub_;
        ros::Subscriber gtPose_sub_, stampedPose_sub_;
        ros::Publisher pose_gt_pub_, path_gt_pub_;
        image_transport::Publisher reprojDepthMap_pub_left_;

        // results
        nav_msgs::Path path_gt_;
        // std::list<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> lPose_GT_;
        // std::list<std::string> lTimestamp_GT_;

        // offline data
        std::string lidar_frame_id_;
        std::string dvs_frame_id_;
        std::string world_frame_id_;
        std::string calibInfoDir_;
        esvo_core::container::CameraSystem::Ptr camSysPtr_;
        esvo_core::container::SensorSystem::Ptr sensorSysPtr_;

        // inter-thread management
        std::mutex data_mutex_;

        // online data
        std::deque<std::pair<ros::Time, pcl::PointCloud<pcl::PointXYZI>::Ptr>> cloud_buf_;
        std::map<ros::Time, Eigen::Matrix4d> mAllPoses_, mAllGTPoses_; // save the historical poses for mapping
        std::vector<std::pair<ros::Time, Eigen::Matrix4d>> mVirtualPoses_;
        esvo_core::tools::LinearTrajectory trajectory_;
        Eigen::Matrix4d T_world_map_;
        std::shared_ptr<tf::Transformer> tf_;

        /**** offline parameters ***/
        size_t depthmap_rate_hz_;
        size_t CLOUD_HISTORY_LENGTH_;
        bool bSaveDepthMap_;
        bool bVisualizeDepthMap_;
        std::string resultPath_;
    };
} // namespace esvo_core

#endif