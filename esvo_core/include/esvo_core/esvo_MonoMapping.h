#ifndef ESVO_CORE_MONOMAPPING_H
#define ESVO_CORE_MONOMAPPING_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <tf2_ros/transform_broadcaster.h>

#include <esvo_core/container/CameraSystem.h>
#include <esvo_core/container/DepthMap.h>
#include <esvo_core/container/TimeSurfaceObservation.h>
#include <esvo_core/core/DepthMonoFusion.h>
#include <esvo_core/core/DepthMonoRegularization.h>
#include <esvo_core/core/DepthProblemConfig.h>
#include <esvo_core/tools/utils.h>
#include <esvo_core/tools/Visualization.h>

#include <dynamic_reconfigure/server.h>
#include <esvo_core/DVS_MappingStereoConfig.h>

#include <emvs_core/MapperEMVS.hpp>
#include <emvs_core/Trajectory.hpp>

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

const double VAR_RANDOM_INIT_INITIAL_ = 0.2;
const double INIT_DP_NUM_Threshold_ = 1000;

namespace esvo_core
{
    using namespace core;
    class esvo_MonoMapping
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        esvo_MonoMapping(const ros::NodeHandle &nh,
                         const ros::NodeHandle &nh_private);
        virtual ~esvo_MonoMapping();

        // mapping
        void MappingLoop(std::promise<void> prom_mapping, std::future<void> future_reset);
        void MappingAtTime(const ros::Time &t);
        bool InitializationAtTime(const ros::Time &t);
        bool dataTransferring();

        // callback functions
        void stampedPoseCallback(const geometry_msgs::PoseStampedConstPtr &ps_msg);
        void eventsCallback(const dvs_msgs::EventArray::ConstPtr &msg, EventQueue &EQ);
        void timeSurfaceCallback(const sensor_msgs::ImageConstPtr &time_surface_left);
        void onlineParameterChangeCallback(DVS_MappingStereoConfig &config, uint32_t level);

        // utils
        void clearEventQueue(EventQueue &EQ);
        void reset();

        /*** publish results ***/
        void publishMappingResults(
            DepthMap::Ptr depthMapPtr,
            Eigen::Matrix4d T,
            ros::Time t);
        void publishPointCloud(
            DepthMap::Ptr &depthMapPtr,
            Eigen::Matrix4d &T,
            ros::Time &t);
        void publishImage(
            const cv::Mat &image,
            const ros::Time &t,
            image_transport::Publisher &pub,
            std::string encoding = "bgr8");
        void saveDepthMap(
            DepthMap::Ptr &depthMapPtr,
            std::string &saveDir,
            ros::Time t);

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

        /************************ member variables ************************/
    private:
        ros::NodeHandle nh_, pnh_;

        // Subscribers
        ros::Subscriber events_left_sub_;
        ros::Subscriber stampedPose_sub_;
        ros::Subscriber TS_left_sub_;

        // Publishers
        ros::Publisher pc_pub_, gpc_pub_;
        image_transport::ImageTransport it_;
        double t_last_pub_pc_;

        // dynamic configuration (modify parameters online)
        boost::shared_ptr<dynamic_reconfigure::Server<DVS_MappingStereoConfig>> server_;
        dynamic_reconfigure::Server<DVS_MappingStereoConfig>::CallbackType dynamic_reconfigure_callback_;

        // offline data
        std::string dvs_frame_id_;
        std::string world_frame_id_;
        std::string calibInfoDir_;
        CameraSystem::Ptr camSysPtr_;

        // online data
        EventQueue events_left_;
        TimeSurfaceHistory TS_history_;
        StampedTimeSurfaceObs TS_obs_;
        StampTransformationMap st_map_;
        std::shared_ptr<tf::Transformer> tf_;
        size_t TS_id_;
        ros::Time tf_lastest_common_time_;

        // system
        std::string ESVO_System_Status_;
        DepthProblemConfig::Ptr dpConfigPtr_;
        DepthMonoFusion dFusor_;
        DepthMonoRegularization dRegularizor_;
        Visualization visualizor_;

        // data transfer
        std::vector<dvs_msgs::Event *> vALLEventsPtr_left_;      
        std::vector<dvs_msgs::Event *> vCloseEventsPtr_left_;    
        std::vector<dvs_msgs::Event *> vDenoisedEventsPtr_left_; 
        size_t totalNumCount_;                                   // count the number of events involved

        // result
        PointCloud::Ptr pc_, pc_near_, pc_global_;
        DepthFrame::Ptr depthFramePtr_;
        std::deque<std::vector<DepthPoint>> dqvDepthPoints_;

        // inter-thread management
        std::mutex data_mutex_;
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
        double stdVar_init_;
        size_t age_max_range_;
        size_t age_vis_threshold_;
        int fusion_radius_;
        std::string FusionStrategy_;
        int maxNumFusionFrames_;
        int maxNumFusionPoints_;
        // module parameters
        size_t PROCESS_EVENT_NUM_;
        size_t TS_HISTORY_LENGTH_;
        size_t mapping_rate_hz_;
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
        // initialization parameters
        bool planarDepthMap_Init_, nonPlanarDepthMap_Init_;

        /**********************************************************/
        /******************** For test & debug ********************/
        /**********************************************************/
        image_transport::Publisher invDepthMap_pub_, stdVarMap_pub_, ageMap_pub_, costMap_pub_;

        // For counting the total number of fusion
        size_t TotalNumFusion_;

        /**********************************************************/
        /******************** EMVS_Mapping ************************/
        /**********************************************************/
        void insertKeyframe();
        void publishDSIResults(const ros::Time &t, const cv::Mat &semiDenseMask,
                               const cv::Mat &depthMap, const cv::Mat &confidenceMap);

        EMVS::ShapeDSI emvs_dsi_shape_;
        EMVS::OptionsDepthMap emvs_opts_depth_map_;
        EMVS::OptionsPointCloud emvs_opts_pc_;
        EMVS::OptionsMapper emvs_opts_mapper_;
        EMVS::MapperEMVS emvs_mapper_;

        std::map<ros::Time, Eigen::Matrix4d> mAllPoses_; // save the historical poses for mapping
        std::vector<std::pair<ros::Time, Eigen::Matrix4d>> mVirtualPoses_;
        EMVS::LinearTrajectory trajectory_;
        bool isKeyframe_;
        Eigen::Matrix4d T_w_keyframe_, T_w_frame_;

        double meanDepth_;
        double KEYFRAME_MEANDEPTH_DIS_;
        image_transport::Publisher depthMap_pub_, confidenceMap_pub_, semiDenseMask_pub_;
        int EMVS_Keyframe_event_, EMVS_Init_event_;

        bool SAVE_RESULT_;
        std::string resultPath_;
        double invDepth_INIT_;
    };
} // namespace esvo_core

#endif //ESVO_CORE_MONOMAPPING_H
