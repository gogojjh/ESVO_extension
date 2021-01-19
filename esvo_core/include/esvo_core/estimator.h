#ifndef ESTIMATOR_H
#define ESTIMATOR_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <tf/transform_broadcaster.h>
#include <std_msgs/Int16.h>

#include <nav_msgs/Path.h>

#include <esvo_core/container/CameraSystem.h>
#include <esvo_core/container/DepthMap.h>
#include <esvo_core/container/EventMatchPair.h>
#include <esvo_core/core/DepthFusion.h>
#include <esvo_core/core/DepthRegularization.h>
#include <esvo_core/core/DepthProblem.h>
#include <esvo_core/core/DepthProblemSolver.h>
#include <esvo_core/core/EventBM.h>
#include <esvo_core/tools/utils.h>
#include <esvo_core/tools/Visualization.h>

#include <esvo_core/core/RegProblemLM.h>
#include <esvo_core/core/RegProblemSolverLM.h>

#include <dynamic_reconfigure/server.h>
#include <esvo_core/DVS_MappingStereoConfig.h>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "emvs_core/MapperEMVS.hpp"
#include "emvs_core/Trajectory.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <map>
#include <deque>
#include <mutex>
#include <future>

#include <cv_bridge/cv_bridge.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>

#define ESTIMATOR_DEBUG
// #define MONOCULAR_DEBUG
// #define EMVS_MAPPING_DEBUG

const double VAR_RANDOM_INIT_INITIAL_ = 0.2;
const double INIT_DP_NUM_Threshold_ = 500;

namespace esvo_core
{
    using namespace core;
    class esvo_Mapping
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        esvo_Mapping(const ros::NodeHandle &nh, const ros::NodeHandle &nh_private);
        virtual ~esvo_Mapping();

        // mapping
        void MappingLoop(std::promise<void> prom_mapping, std::future<void> future_reset);
        void Process();
        bool InitializationAtTime(const ros::Time &t);
        bool MonoInitializationAtTime(const ros::Time &t);
        void MappingAtTime(const ros::Time &t);
        void MonoMappingAtTime(const ros::Time &t);
        void insertKeyframe();
        bool dataTransferring();

        // callback functions
        void stampedPoseCallback(const geometry_msgs::PoseStampedConstPtr &ps_msg);
        void eventsCallback(const dvs_msgs::EventArray::ConstPtr &msg, EventQueue &EQ);
        void timeSurfaceCallback(const sensor_msgs::ImageConstPtr &time_surface_left,
                                 const sensor_msgs::ImageConstPtr &time_surface_right);
        void onlineParameterChangeCallback(DVS_MappingStereoConfig &config, uint32_t level);

        // utils
        bool getPoseAt(const ros::Time &t, Transformation &Tr, const std::string &source_frame);
        void clearEventQueue(EventQueue &EQ);
        void reset();

        /*** publish results ***/
        void publishMappingResults(DepthMap::Ptr depthMapPtr, Transformation tr, ros::Time t);
        void publishPointCloud(
            DepthMap::Ptr &depthMapPtr,
            Transformation &tr,
            ros::Time &t);

        void publishDSIResults(const ros::Time &t, const cv::Mat &depthMap, const cv::Mat &semiDenseMask);

        void publishEventMap(const ros::Time &t);

        void publishImage(
            const cv::Mat &image,
            const ros::Time &t,
            image_transport::Publisher &pub,
            std::string encoding = "bgr8");

        /*** event processing ***/
        void createEdgeMask(
            std::vector<dvs_msgs::Event *> &vEventsPtr,
            PerspectiveCamera::Ptr &camPtr,
            cv::Mat &edgeMap,
            std::vector<std::pair<size_t, size_t>> &vEdgeletCoordinates,
            bool bUndistortEvents = true,
            size_t radius = 0);

        void createDenoisingMask(
            std::vector<dvs_msgs::Event *> &vAllEventsPtr,
            cv::Mat &mask,
            size_t row, size_t col); // reserve in this file

        void extractDenoisedEvents(
            std::vector<dvs_msgs::Event *> &vCloseEventsPtr,
            std::vector<dvs_msgs::Event *> &vEdgeEventsPtr,
            cv::Mat &mask,
            size_t maxNum = 5000);

        // results
        void publishPose(const ros::Time &t, Transformation &tr);
        void publishPath(const ros::Time &t, Transformation &tr);
        void saveTrajectory(const std::string &resultDir);

        /************************ member variables ************************/
    private:
        ros::NodeHandle nh_, pnh_;

        // Subscribers
        ros::Subscriber events_left_sub_, events_right_sub_;
        ros::Subscriber stampedPose_sub_;
        message_filters::Subscriber<sensor_msgs::Image> TS_left_sub_, TS_right_sub_;
        ros::Subscriber keyFrame_sub_;

        // Publishers
        ros::Publisher pc_pub_, gpc_pub_;
        image_transport::ImageTransport it_;
        double t_last_pub_pc_;

        // Time-Surface sync policy
        typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image> ExactSyncPolicy;
        message_filters::Synchronizer<ExactSyncPolicy> TS_sync_;

        // dynamic configuration (modify parameters online)
        boost::shared_ptr<dynamic_reconfigure::Server<DVS_MappingStereoConfig>> server_;
        dynamic_reconfigure::Server<DVS_MappingStereoConfig>::CallbackType dynamic_reconfigure_callback_;

        // offline data
        std::string dvs_frame_id_;
        std::string world_frame_id_;
        std::string calibInfoDir_;
        CameraSystem::Ptr camSysPtr_;

        // online data
        EventQueue events_left_, events_right_;
        TimeSurfaceHistory TS_history_;
        StampedTimeSurfaceObs TS_obs_;
        StampTransformationMap st_map_;
        std::shared_ptr<tf::Transformer> tf_;
        size_t TS_id_;
        ros::Time tf_lastest_common_time_;

        std::deque<std::pair<ros::Time, TimeSurfaceObservation>> TS_buf_;

        // system
        std::string ESVO_System_Status_;
        DepthProblemConfig::Ptr dpConfigPtr_;
        DepthProblemSolver dpSolver_;
        DepthFusion dFusor_;
        DepthRegularization dRegularizor_;
        Visualization visualizor_;
        EventBM ebm_;

        // data transfer
        std::vector<dvs_msgs::Event *> vALLEventsPtr_left_;      // for BM
        std::vector<dvs_msgs::Event *> vCloseEventsPtr_left_;    // for BM
        std::vector<dvs_msgs::Event *> vDenoisedEventsPtr_left_; // for BM
        size_t totalNumCount_;                                   // count the number of events involved
        std::vector<dvs_msgs::Event *> vEventsPtr_left_SGM_;     // for SGM

        // result
        PointCloud::Ptr pc_; // local depth map
        PointCloud::Ptr pc_near_; // local depth map within the viable range
        PointCloud::Ptr pc_global_; // global depth map which is insert into the filtered pc_near_
        DepthFrame::Ptr depthFramePtr_;
        std::deque<std::vector<DepthPoint>> dqvDepthPoints_;

        // inter-thread management
        std::mutex data_mutex_;
        std::mutex m_buf_;
        std::promise<void> mapping_thread_promise_, reset_promise_;
        std::future<void> mapping_thread_future_, reset_future_;

        /**** mapping parameters ***/
        // range and visualization threshold
        double invDepth_min_range_;
        double invDepth_max_range_;
        double cost_vis_threshold_;
        size_t patch_area_;
        double residual_vis_threshold_;
        double stdVar_vis_threshold_;
        size_t age_max_range_;
        size_t age_vis_threshold_;
        int fusion_radius_;
        std::string FusionStrategy_;
        int maxNumFusionFrames_;
        int maxNumFusionPoints_;
        size_t INIT_SGM_DP_NUM_Threshold_;
        // module parameters
        size_t PROCESS_EVENT_NUM_;
        size_t TS_HISTORY_LENGTH_;
        size_t mapping_rate_hz_;
        size_t process_rate_hz_;
        // options
        bool changed_frame_rate_;
        bool bRegularization_;
        bool resetButton_;
        bool bDenoising_;
        bool bVisualizeGlobalPC_;
        // visualization parameters
        double visualizeGPC_interval_;
        double visualize_range_;
        size_t numAddedPC_threshold_;
        // Event Block Matching (BM) parameters
        double BM_half_slice_thickness_;
        size_t BM_patch_size_X_;
        size_t BM_patch_size_Y_;
        size_t BM_min_disparity_;
        size_t BM_max_disparity_;
        size_t BM_step_;
        double BM_ZNCC_Threshold_;
        bool BM_bUpDownConfiguration_;

        // SGM parameters (Used by Initialization)
        int num_disparities_;
        int block_size_;
        int P1_;
        int P2_;
        int uniqueness_ratio_;
        cv::Ptr<cv::StereoSGBM> sgbm_;

        /**********************************************************/
        /******************** For test & debug ********************/
        /**********************************************************/
        image_transport::Publisher invDepthMap_pub_, stdVarMap_pub_, ageMap_pub_, costMap_pub_;
        image_transport::Publisher depthMap_pub_, confidenceMap_pub_;
        image_transport::Publisher eventMap_pub_;

        // For counting the total number of fusion
        size_t TotalNumFusion_;

        double invDepth_INIT_;

        camodocal::CameraPtr camPtr_, camVirtualPtr_;
        EMVS::MapperEMVS emvs_mapper_;
        EMVS::ShapeDSI emvs_dsi_shape_;
        EMVS::OptionsDepthMap emvs_opts_depth_map_;
        std::map<ros::Time, Eigen::Matrix4d> mAllPoses_;
        std::vector<std::pair<ros::Time, Eigen::Matrix4d>> mVirtualPoses_;
        EMVS::LinearTrajectory trajectory_;
        bool isKeyframe_;
    };
} // namespace esvo_core

#endif //ESTIMATOR_H
