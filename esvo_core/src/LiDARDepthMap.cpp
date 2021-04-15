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
        CLOUD_HISTORY_LENGTH_ = tools::param(pnh_, "CLOUD_HISTORY_LENGTH", 20);
        CLOUD_MAP_LENGTH_ = tools::param(pnh_, "CLOUD_MAP_LENGTH", 10);
        CLOUD_RES_ = tools::param(pnh_, "CLOUD_RES", 0.2);
        downSizeFilter_.setLeafSize(CLOUD_RES_, CLOUD_RES_, CLOUD_RES_);
        KEYFRAME_th_dis_ = tools::param(pnh_, "KEYFRAME_th_dis", 0.1);
        KEYFRAME_th_ori_ = tools::param(pnh_, "KEYFRAME_th_ori", 3);

        Eigen::Vector3d p0;
        camSysPtr_->cam_left_ptr_->cam2World(Eigen::Vector2d(0.0, 0.0), 
                                             1.0, 
                                             p0);
        Eigen::Vector3d p1;
        camSysPtr_->cam_left_ptr_->cam2World(Eigen::Vector2d(camSysPtr_->cam_left_ptr_->width_, camSysPtr_->cam_left_ptr_->height_),
                                             1.0, 
                                             p1);
        camFOV_x_ = atan2(p1(0) - p0(0), 1.0) / M_PI * 180.0;
        camFOV_y_ = atan2(p1(1) - p0(1), 1.0) / M_PI * 180.0;
        FOV_ratio_X_ = tools::param(pnh_, "FOV_ratio_X", 0.5);
        FOV_ratio_Y_ = tools::param(pnh_, "FOV_ratio_Y", 0.5);
        LOG(INFO) << "camFOV_x: " << camFOV_x_ << ", camFOV_y: " << camFOV_y_;
        LOG(INFO) << "FOV_ratio_X: " << FOV_ratio_X_ << ", FOV_ratio_Y: " << FOV_ratio_Y_;

        /**** online parameters ***/
        depthmap_rate_hz_ = tools::param(pnh_, "depthmap_rate_hz", 10);
        bSaveDepthMap_ = tools::param(pnh_, "SAVE_DEPTH_MAP", false);
        resultPath_ = tools::param(pnh_, "PATH_TO_SAVE_DEPTH_MAP", std::string());
        bVisualizeDepthMap_ = tools::param(pnh_, "VISUALIZE_DEPTH_MAP", true);

        // online data callbacks
        cloud_sub_ = nh_.subscribe("pointcloud", 10, &LiDARDepthMap::pointCloudCallback, this);
        stampedPose_sub_ = nh_.subscribe("stamped_pose", 0, &LiDARDepthMap::stampedPoseCallback, this); // for accessing the pose of the ref view.
        gtPose_sub_ = nh_.subscribe("gt_pose", 0, &LiDARDepthMap::gtPoseCallback, this);                // for accessing the pose of the ref view.

#ifdef PUBLISH_PATH
        pose_gt_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/gt/pose_pub", 1);
        path_gt_pub_ = nh_.advertise<nav_msgs::Path>("/gt/trajectory", 1);
#endif        
        depthMap_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/lidardepthmap/point_cloud", 1);

        tf_ = std::make_shared<tf::Transformer>(true, ros::Duration(100.0));

        // online data
        cloudHistory_.clear();
        mAllPoses_.clear();
        mAllGTPoses_.clear();
        T_world_map_.setIdentity();
        cur_.depthMap_ptr_.reset(new PointCloudI());
        cur_.stampedCloudPose_.clear();

        /*** For Visualization and Test ***/
        reprojDepthMap_pub_left_ = it_.advertise("reprojDepthMap_left", 1);

        /*** LiDAR Depth Map ***/
        std::thread DepthMapThread(&LiDARDepthMap::DepthMapLoop, this);
        DepthMapThread.detach();
    }

    LiDARDepthMap::~LiDARDepthMap()
    {
#ifdef PUBLISH_PATH
        path_gt_pub_.shutdown();
        pose_gt_pub_.shutdown();
#endif
    }

    void LiDARDepthMap::reset()
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        cloudHistory_.clear();
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
            if (cloudHistory_.size() < 1)
            {
                r.sleep();
                continue;
            }

            {
                std::lock_guard<std::mutex> lock(data_mutex_);
                if (cur_.t_.toSec() < cloudHistory_.back().first.toSec()) // new pointcloud arrives
                {
                    if (!curDataTransferring())
                    {
                        continue;
                        r.sleep();
                    }
                } 
                else
                {
                    continue;
                    r.sleep();
                }

                cur_.depthMap_ptr_->clear();
                auto scp_it = cur_.stampedCloudPose_.begin();
                for (; scp_it != cur_.stampedCloudPose_.end(); scp_it++)
                {
                    for (const pcl::PointXYZI &point : *scp_it->first.second)
                    {
                        pcl::PointXYZI un_point;
                        TransformToStart(point, un_point, scp_it->second); // linear interpolation according to time
                        cur_.depthMap_ptr_->push_back(un_point);
                    }
                }
#ifdef LIDAR_DEPTH_MAP_DEBUG
                LOG_EVERY_N(INFO, 100) << "Size of depth map: " << cur_.depthMap_ptr_->size();
#endif
                publishPointCloud(cur_.t_, cur_.depthMap_ptr_, depthMap_pub_);
            }
        } // while

    }

    bool LiDARDepthMap::curDataTransferring()
    {
        if (cur_.t_.toSec() == cloudHistory_.back().first.toSec())
            return false;

        auto cloud_it = cloudHistory_.rbegin();
        Eigen::Matrix4d T_w_pc;
        while (cloud_it != cloudHistory_.rend()) // append the latest pointcloud
        {
            // if (trajectory_.getPoseAt(mAllGTPoses_, cloud_it->first, T_w_pc)) // check if poses are ready
            // {
            //     break;
            // }
            if (trajectory_.getPoseAt(mAllPoses_, cloud_it->first, T_w_pc)) // check if poses are ready
            {
                break;
            }
            cloud_it++;
        }
        if (cloud_it == cloudHistory_.rend())
            return false;
        
        if (!cur_.stampedCloudPose_.empty())
        {
            const Eigen::Matrix4d &T_w_pc0 = cur_.stampedCloudPose_.back().second;
            Eigen::Quaterniond q_w_pc0(T_w_pc0.topLeftCorner<3, 3>());
            Eigen::Quaterniond q_w_pc(T_w_pc.topLeftCorner<3, 3>());
            double ang = q_w_pc0.angularDistance(q_w_pc) / M_PI * 180.0;
            double dis = (T_w_pc0.topRightCorner<3, 1>() - T_w_pc.topRightCorner<3, 1>()).norm();
            if (ang < KEYFRAME_th_ori_ && dis < KEYFRAME_th_dis_)
                return false;
        }

        cur_.t_ = cloudHistory_.back().first;

        TicToc t_add_point;
        // transform LiDAR point cloud onto left davis frame, and keep points within cameras' FOV
        PointCloudI::Ptr pcTrans_ptr(new PointCloudI());
        for (auto &p : *cloudHistory_.back().second)
        {
            Eigen::Vector3d point(p.x, p.y, p.z);
            Eigen::Vector3d pointCam = sensorSysPtr_->T_left_davis_lidar_.topLeftCorner<3, 3>() * point + 
                                       sensorSysPtr_->T_left_davis_lidar_.topRightCorner<3, 1>();
            double ang_x = std::abs(atan2(pointCam(0), pointCam(2))) / M_PI * 180.0;
            double ang_y = std::abs(atan2(pointCam(1), pointCam(2))) / M_PI * 180.0;
            if (ang_x > camFOV_x_ * FOV_ratio_X_ || ang_y > camFOV_y_ * FOV_ratio_Y_)
                continue;
            auto pCam = p;
            pCam.x = pointCam(0);
            pCam.y = pointCam(1);
            pCam.z = pointCam(2);
            pcTrans_ptr->push_back(pCam);
        }
        
        RefPointCloudIPair refPointCloudIPair = std::make_pair(cur_.t_, pcTrans_ptr);
        while (cur_.stampedCloudPose_.size() >= CLOUD_MAP_LENGTH_) // default: 10
            cur_.stampedCloudPose_.pop_front();
        cur_.stampedCloudPose_.emplace_back(refPointCloudIPair, T_w_pc);
        LOG_EVERY_N(INFO, 20) << t_add_point.toc() << "ms";
    }

    /********************** Callback functions *****************************/
    void LiDARDepthMap::pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        PointCloudI::Ptr PC_ptr(new PointCloudI());
        pcl::fromROSMsg(*msg, *PC_ptr);
        cloudHistory_.emplace_back(msg->header.stamp, PC_ptr);
        // PointCloudI::Ptr PC_ds_ptr(new PointCloudI());
        // downSizeFilter_.setInputCloud(PC_ptr);
        // downSizeFilter_.filter(*PC_ds_ptr);
        // cloudHistory_.emplace_back(msg->header.stamp, PC_ds_ptr);
        while (cloudHistory_.size() > CLOUD_HISTORY_LENGTH_) // 20
            cloudHistory_.pop_front();
    }

    void LiDARDepthMap::stampedPoseCallback(const geometry_msgs::PoseStampedConstPtr &ps_msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);

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

#ifdef PUBLISH_PATH
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
#endif
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

#ifdef PUBLISH_PATH
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
#endif 
    }

    void LiDARDepthMap::publishPointCloud(const ros::Time &t,
                                          const PointCloudI::Ptr &pc_ptr,
                                          const ros::Publisher &pc_pub)
    {
        sensor_msgs::PointCloud2::Ptr pc_to_publish(new sensor_msgs::PointCloud2);
        // Eigen::Matrix<double, 4, 4> T_world_result = tr.getTransformationMatrix();

        // double FarthestDistance = 0.0;
        // Eigen::Vector3d FarthestPoint;

        // for (auto it = depthMapPtr->begin(); it != depthMapPtr->end(); it++)
        // {
        //     Eigen::Vector3d p_world = T_world_result.block<3, 3>(0, 0) * it->p_cam() + T_world_result.block<3, 1>(0, 3);
        //     pc_->push_back(pcl::PointXYZ(p_world(0), p_world(1), p_world(2)));

        //     if (it->p_cam().norm() < visualize_range_)
        //         pc_near_->push_back(pcl::PointXYZ(p_world(0), p_world(1), p_world(2)));
        //     if (it->p_cam().norm() > FarthestDistance)
        //     {
        //         FarthestDistance = it->p_cam().norm();
        //         FarthestPoint = it->p_cam();
        //     }
        // }

        if (!pc_ptr->empty())
        {
            pcl::toROSMsg(*pc_ptr, *pc_to_publish);
            pc_to_publish->header.stamp = t;
            pc_to_publish->header.frame_id = world_frame_id_.c_str();
            pc_pub.publish(pc_to_publish);
        }

        // // publish global pointcloud
        // if (bVisualizeGlobalPC_)
        // {
        //     if (t.toSec() - t_last_pub_pc_ > visualizeGPC_interval_)
        //     {
        //         PointCloud::Ptr pc_filtered(new PointCloud());
        //         pcl::VoxelGrid<pcl::PointXYZ> sor;
        //         sor.setInputCloud(pc_near_);
        //         sor.setLeafSize(0.03, 0.03, 0.03);
        //         sor.filter(*pc_filtered);

        //         // copy the most current pc tp pc_global
        //         size_t pc_length = pc_filtered->size();
        //         size_t numAddedPC = min(pc_length, numAddedPC_threshold_) - 1;
        //         pc_global_->insert(pc_global_->end(), pc_filtered->end() - numAddedPC, pc_filtered->end());

        //         // publish point cloud
        //         pcl::toROSMsg(*pc_global_, *pc_to_publish);
        //         pc_to_publish->header.stamp = t;
        //         gpc_pub_.publish(pc_to_publish);
        //         t_last_pub_pc_ = t.toSec();
        //     }
        // }
    }

    void LiDARDepthMap::saveDepthMap(const std::string &resultDir,
                                    const PointCloudI::Ptr &pc_ptr)

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
