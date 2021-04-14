#include <esvo_core/LiDARDepthMap.h>
#include <esvo_core/tools/TicToc.h>
#include <esvo_core/tools/params_helper.h>
#include <tf/transform_broadcaster.h>
#include <sys/stat.h>

namespace esvo_core
{
    LiDARDepthMap::LiDARDepthMap(
        const ros::NodeHandle &nh,
        const ros::NodeHandle &nh_private) : nh_(nh),
                                             pnh_(nh_private),
                                             it_(nh),
                                             calibInfoDir_(tools::param(pnh_, "calibInfoDir", std::string(""))),
                                             camSysPtr_(new CameraSystem(calibInfoDir_, false)),
                                             sensorSysPtr_(new SensorSystem(calibInfoDir_, true))
    {
        // offline data
        lidar_frame_id_ = tools::param(pnh_, "lidar_frame_id", std::string("velodyne"));
        dvs_frame_id_ = tools::param(pnh_, "dvs_frame_id", std::string("dvs"));
        world_frame_id_ = tools::param(pnh_, "world_frame_id", std::string("world"));
        CLOUD_HISTORY_LENGTH_ = tools::param(pnh_, "CLOUD_HISTORY_LENGTH", 10);

        /**** online parameters ***/
        depthmap_rate_hz_ = tools::param(pnh_, "depthmap_rate_hz", 10);
        bSaveDepthMap_ = tools::param(pnh_, "SAVE_DEPTH_MAP", false);
        resultPath_ = tools::param(pnh_, "PATH_TO_SAVE_DEPTH_MAP", std::string());
        bVisualizeDepthMap_ = tools::param(pnh_, "VISUALIZE_DEPTH_MAP", true);

        // online data callbacks
        cloud_sub_ = nh_.subscribe("pointcloud", 10, &LiDARDepthMap::pointCloudCallback, this);
        stampedPose_sub_ = nh_.subscribe("stamped_pose", 0, &LiDARDepthMap::stampedPoseCallback, this); // for accessing the pose of the ref view.
        gtPose_sub_ = nh_.subscribe("gt_pose", 0, &LiDARDepthMap::gtPoseCallback, this);                // for accessing the pose of the ref view.

        pose_gt_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/gt/pose_pub", 1);
        path_gt_pub_ = nh_.advertise<nav_msgs::Path>("/gt/trajectory", 1);

        tf_ = std::make_shared<tf::Transformer>(true, ros::Duration(100.0));

        // online data
        cloud_buf_.clear();
        mAllPoses_.clear();
        mAllGTPoses_.clear();
        T_world_map_.setIdentity();

        /*** For Visualization and Test ***/
        reprojDepthMap_pub_left_ = it_.advertise("reprojDepthMap_left", 1);

        /*** LiDAR Depth Map ***/
        std::thread DepthMapThread(&LiDARDepthMap::DepthMapLoop, this);
        DepthMapThread.detach();
    }

    LiDARDepthMap::~LiDARDepthMap()
    {
        path_gt_pub_.shutdown();
        pose_gt_pub_.shutdown();
    }

    void LiDARDepthMap::reset()
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        cloud_buf_.clear();
        mAllPoses_.clear();
        mAllGTPoses_.clear();
        T_world_map_.setIdentity();

        path_gt_.poses.clear();
    }

    void LiDARDepthMap::DepthMapLoop()
    {
        ros::Rate r(depthmap_rate_hz_);
        while (ros::ok())
        {
            // Keep Idling
            if (cloud_buf_.size() < 1)
            {
                r.sleep();
                continue;
            }
            r.sleep();
        } // while

        // if (trajectory_.getPoseAt(mAllPoses_, it_end->first, T_w_obs)) // check if poses are ready
        // {
        //     it_end->second.setTransformation(T_w_obs);
        //     TS_obs_ = *it_end;
        // }
    }

    /********************** Callback functions *****************************/
    void LiDARDepthMap::pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        pcl::PointCloud<pcl::PointXYZI>::Ptr PC_ptr(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::fromROSMsg(*msg, *PC_ptr);
        // transform LiDAR point cloud onto left camera frame
        pcl::transformPointCloud(*PC_ptr, *PC_ptr, sensorSysPtr_->T_left_davis_lidar_.cast<float>());
        cloud_buf_.emplace_back(msg->header.stamp, PC_ptr);
        while (cloud_buf_.size() > CLOUD_HISTORY_LENGTH_) // 20
            cloud_buf_.pop_front();
    }

    void LiDARDepthMap::stampedPoseCallback(const geometry_msgs::PoseStampedConstPtr &ps_msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        // add pose to tf
        tf::Transform tf(
            tf::Quaternion(
                ps_msg->pose.orientation.x,
                ps_msg->pose.orientation.y,
                ps_msg->pose.orientation.z,
                ps_msg->pose.orientation.w),
            tf::Vector3(
                ps_msg->pose.position.x,
                ps_msg->pose.position.y,
                ps_msg->pose.position.z));
        tf::StampedTransform st(tf, ps_msg->header.stamp, ps_msg->header.frame_id, dvs_frame_id_.c_str());
        tf_->setTransform(st);
        // broadcast the tf such that the nav_path messages can find the valid fixed frame "map".
        static tf::TransformBroadcaster br;
        br.sendTransform(st);

        Eigen::Matrix4d T_map_cam = Eigen::Matrix4d::Identity();
        T_map_cam.topLeftCorner<3, 3>() = Eigen::Quaterniond(ps_msg->pose.orientation.w,
                                                            ps_msg->pose.orientation.x,
                                                            ps_msg->pose.orientation.y,
                                                            ps_msg->pose.orientation.z)
                                            .toRotationMatrix();
        T_map_cam.topRightCorner<3, 1>() = Eigen::Vector3d(ps_msg->pose.position.x,
                                                        ps_msg->pose.position.y,
                                                        ps_msg->pose.position.z);
        mAllPoses_.emplace(ps_msg->header.stamp, T_map_cam);
        static constexpr size_t MAX_POSE_LENGTH = 10000; // the buffer size
        if (mAllPoses_.size() > 20000)
        {
            size_t removeCnt = 0;
            for (auto it_pose = mAllPoses_.begin(); it_pose != mAllPoses_.end();)
            {
                mAllPoses_.erase(it_pose++);
                removeCnt++;
                if (removeCnt >= mAllPoses_.size() - MAX_POSE_LENGTH)
                    break;
            }
        }
    }

    void LiDARDepthMap::gtPoseCallback(const geometry_msgs::PoseStampedConstPtr &ps_msg)
    {
        Eigen::Matrix4d T_world_marker = Eigen::Matrix4d::Identity();
        T_world_marker.topLeftCorner<3, 3>() = Eigen::Quaterniond(ps_msg->pose.orientation.w,
                                                                ps_msg->pose.orientation.x,
                                                                ps_msg->pose.orientation.y,
                                                                ps_msg->pose.orientation.z)
                                                .toRotationMatrix();
        T_world_marker.topRightCorner<3, 1>() = Eigen::Vector3d(ps_msg->pose.position.x,
                                                                ps_msg->pose.position.y,
                                                                ps_msg->pose.position.z);

        // HARDCODED: The GT pose of rpg dataset is the pose of stereo rig, namely that of the marker.
        Eigen::Matrix4d T_marker_cam = sensorSysPtr_->T_gt_left_davis_;
        Eigen::Matrix4d T_world_cam = T_world_marker * T_marker_cam;
        if (T_world_map_ == Eigen::Matrix4d::Identity())
            T_world_map_ = T_world_cam;
        Eigen::Matrix4d T_map_cam = T_world_map_.inverse() * T_world_cam;
        Eigen::Matrix3d R_map_cam = T_map_cam.topLeftCorner<3, 3>();
        Eigen::Quaterniond q_map_cam(R_map_cam);
        Eigen::Vector3d t_map_cam = T_map_cam.topRightCorner<3, 1>();

        tf::Transform tf(
            tf::Quaternion(
                q_map_cam.x(),
                q_map_cam.y(),
                q_map_cam.z(),
                q_map_cam.w()),
            tf::Vector3(
                t_map_cam.x(),
                t_map_cam.y(),
                t_map_cam.z()));
        tf::StampedTransform st(tf, ps_msg->header.stamp, world_frame_id_, std::string(dvs_frame_id_ + "_gt").c_str());
        tf_->setTransform(st);

        // broadcast the tf such that the nav_path messages can find the valid fixed frame "map".
        static tf::TransformBroadcaster br;
        br.sendTransform(st);

        // set published gt pose
        geometry_msgs::PoseStampedPtr ps_ptr(new geometry_msgs::PoseStamped());
        ps_ptr->header.stamp = ps_msg->header.stamp;
        ps_ptr->header.frame_id = world_frame_id_;
        ps_ptr->pose.orientation.x = q_map_cam.x();
        ps_ptr->pose.orientation.y = q_map_cam.y();
        ps_ptr->pose.orientation.z = q_map_cam.z();
        ps_ptr->pose.orientation.w = q_map_cam.w();
        ps_ptr->pose.position.x = t_map_cam.x();
        ps_ptr->pose.position.y = t_map_cam.y();
        ps_ptr->pose.position.z = t_map_cam.z();
        pose_gt_pub_.publish(ps_ptr);

        path_gt_.header.stamp = ps_msg->header.stamp;
        path_gt_.header.frame_id = world_frame_id_;
        path_gt_.poses.push_back(*ps_ptr);
        path_gt_pub_.publish(path_gt_);

        mAllGTPoses_.emplace(ps_msg->header.stamp, T_map_cam);
        static constexpr size_t MAX_POSE_LENGTH = 10000; // the buffer size
        if (mAllGTPoses_.size() > 20000)
        {
            size_t removeCnt = 0;
            for (auto it_pose = mAllGTPoses_.begin(); it_pose != mAllGTPoses_.end();)
            {
                mAllGTPoses_.erase(it_pose++);
                removeCnt++;
                if (removeCnt >= mAllGTPoses_.size() - MAX_POSE_LENGTH)
                    break;
            }
        }
    }

    void LiDARDepthMap::saveDepthMap(const std::string &resultDir,
                                    const pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_ptr)

    {
    #ifdef ESVO_CORE_TRACKING_LOG
        LOG(INFO) << "Saving trajectory to " << resultDir << " ......";
    #endif
        pcl::io::savePCDFile(resultDir, *pc_ptr);
    //     std::ofstream f;
    //     f.open(resultDir.c_str(), std::ofstream::out);
    //     if (!f.is_open())
    //     {
    //         LOG(INFO) << "File at " << resultDir << " is not opened, save trajectory failed.";
    //         exit(-1);
    //     }
    //     f << std::fixed;

    //     std::list<Eigen::Matrix<double, 4, 4>,
    //               Eigen::aligned_allocator<Eigen::Matrix<double, 4, 4>>>::const_iterator result_it_begin = lPose.begin();
    //     std::list<Eigen::Matrix<double, 4, 4>,
    //               Eigen::aligned_allocator<Eigen::Matrix<double, 4, 4>>>::const_iterator result_it_end = lPose.end();
    //     std::list<std::string>::const_iterator ts_it_begin = lTimestamp.begin();

    //     for (; result_it_begin != result_it_end; result_it_begin++, ts_it_begin++)
    //     {
    //         Eigen::Matrix3d Rwc_result;
    //         Eigen::Vector3d twc_result;
    //         Rwc_result = (*result_it_begin).block<3, 3>(0, 0);
    //         twc_result = (*result_it_begin).block<3, 1>(0, 3);
    //         Eigen::Quaterniond q(Rwc_result);
    //         f << *ts_it_begin << " " << std::setprecision(9)
    //           << twc_result.x() << " " << twc_result.y() << " " << twc_result.z() << " "
    //           << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    //     }
    //     f.close();
    // #ifdef ESVO_CORE_TRACKING_LOG
    //     LOG(INFO) << "Saving trajectory to " << resultDir << ". Done !!!!!!.";
    // #endif
    }
}
