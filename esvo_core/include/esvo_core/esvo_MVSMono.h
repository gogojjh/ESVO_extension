#ifndef ESVO_CORE_ESVO_MVSMONO_H
#define ESVO_CORE_ESVO_MVSMONO_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <tf/transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include <esvo_core/container/CameraSystem.h>
#include <esvo_core/container/DepthMap.h>
#include <esvo_core/container/EventMatchPair.h>
#include <esvo_core/container/TimeSurfaceObservation.h>
#include <esvo_core/core/DepthFusion.h>
#include <esvo_core/core/DepthMonoFusion.h>
#include <esvo_core/core/DepthRegularization.h>
#include <esvo_core/core/DepthMonoRegularization.h>
#include <esvo_core/core/DepthProblem.h>
#include <esvo_core/core/DepthProblemSolver.h>
#include <esvo_core/core/EventBM.h>
#include <esvo_core/core/EventMatcher.h>
#include <esvo_core/tools/Visualization.h>
#include <esvo_core/tools/utils.h>
#include <esvo_core/DVS_MappingStereoConfig.h>
#include <dynamic_reconfigure/server.h>

#include <emvs_core/MapperEMVS.hpp>
#include <emvs_core/Trajectory.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <deque>
#include <map>
#include <mutex>
#include <future>

#include <boost/filesystem.hpp>

#include <cv_bridge/cv_bridge.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>

#define ESVO_MVSMONO_TRACKING_DEBUG

namespace esvo_core
{
	using namespace core;
	enum eMVSMonoMode
	{
		PURE_EVENT_MATCHING,	   //0 (this one implements GTS [26])
		PURE_BLOCK_MATCHING,	   //1 (this one implements BM)
		EM_PLUS_ESTIMATION,		   //2 (GTS [26] + nonliear opt.)
		BM_PLUS_ESTIMATION,		   //3 (this one is ESVO's mapping method, namely BM + nonliear opt.)
		PURE_SEMI_GLOBAL_MATCHING, //4 (this one implements SGM [45])
		PURE_EMVS,				   //5 (this one implements EMVS [33])
		PURE_EMVS_PLUS_ESTIMATION  //6 (this one implements EMVS [33] + nonliear opt.)
	};

	class esvo_MVSMono
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		esvo_MVSMono(
			const ros::NodeHandle &nh,
			const ros::NodeHandle &nh_private);
		virtual ~esvo_MVSMono();

		// functions regarding mapping
		void MappingLoop(std::promise<void> prom_mapping, std::future<void> future_reset);
		void MappingAtTime(const ros::Time &t);
		bool dataTransferring();

		// callback functions
		void stampedPoseCallback(const geometry_msgs::PoseStampedConstPtr &ps_msg);
		void eventsCallback(const dvs_msgs::EventArray::ConstPtr &msg, EventQueue &EQ);
		void timeSurfaceCallback(const sensor_msgs::ImageConstPtr &time_surface_left);
		void onlineParameterChangeCallback(DVS_MappingStereoConfig &config, uint32_t level);

		// utils
		bool getPoseAt(const ros::Time &t, Transformation &Tr, const std::string &source_frame);
		void clearEventQueue(EventQueue &EQ);
		void reset();

		// results
		void publishMappingResults(
			DepthMap::Ptr depthMapPtr,
			Transformation tr,
			ros::Time t);
		void publishPointCloud(
			DepthMap::Ptr &depthMapPtr,
			Transformation &tr,
			ros::Time &t);
		void publishImage(
			const cv::Mat &image,
			const ros::Time &t,
			image_transport::Publisher &pub,
			std::string encoding = "bgr8");
		void publishKFPose(const ros::Time &t, Transformation &tr);
		void saveDepthMap(
			DepthMap::Ptr &depthMapPtr,
			std::string &saveDir,
			ros::Time t);

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
			size_t row, size_t col);

		void extractDenoisedEvents(
			std::vector<dvs_msgs::Event *> &vCloseEventsPtr,
			std::vector<dvs_msgs::Event *> &vEdgeEventsPtr,
			cv::Mat &mask,
			size_t maxNum = 5000);

	private:
		ros::NodeHandle nh_, pnh_;

		// Subcribers
		ros::Subscriber events_left_sub_, events_right_sub_;
		ros::Subscriber stampedPose_sub_;
		ros::Subscriber TS_left_sub_;
		// message_filters::Subscriber<sensor_msgs::Image> TS_left_sub_, TS_right_sub_;

		// Publishers
		ros::Publisher pc_pub_, gpc_pub_;
		image_transport::ImageTransport it_;

		// Time-Surface sync policy
		// typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image> ExactSyncPolicy;
		// message_filters::Synchronizer<ExactSyncPolicy> TS_sync_;

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

		// system
		eMVSMonoMode msm_;
		DepthProblemConfig::Ptr dpConfigPtr_;
		DepthProblemSolver dpSolver_;
		// DepthFusion dFusor_;
		DepthMonoFusion dFusor_;
		// DepthRegularization dRegularizor_;
		DepthMonoRegularization dRegularizor_;
		Visualization visualizor_;
		EventMatcher em_;
		EventBM ebm_;

		// data transfer
		std::vector<dvs_msgs::Event *> vEventsPtr_left_, vEventsPtr_right_; // for EM
		ros::Time t_lowBound_, t_upBound_;									// for EM
		std::vector<dvs_msgs::Event *> vALLEventsPtr_left_;					// for BM
		std::vector<dvs_msgs::Event *> vCloseEventsPtr_left_;				// for BM
		std::vector<dvs_msgs::Event *> vDenoisedEventsPtr_left_;			// for BM
		size_t totalNumCount_;												// for both
		std::vector<dvs_msgs::Event *> vEventsPtr_left_SGM_;				// for SGM

		// result
		PointCloud::Ptr pc_, pc_global_;
		DepthFrame::Ptr depthFramePtr_;
		std::deque<std::vector<DepthPoint>> dqvDepthPoints_;
		bool bVisualizeGlobalPC_;
		double visualizeGPC_interval_;
		size_t numAddedPC_threshold_;
		double t_last_pub_pc_;

		// inter-thread management
		std::mutex data_mutex_;
		std::promise<void> mapping_thread_promise_, reset_promise_;
		std::future<void> mapping_thread_future_, reset_future_;

		/**** MVStereo parameters ***/
		// range and visualization parameters
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

		// Event Matching (EM [26]) parameters
		double EM_Slice_Thickness_;
		double EM_Time_THRESHOLD_;
		double EM_EPIPOLAR_THRESHOLD_;
		double EM_TS_NCC_THRESHOLD_;
		size_t EM_patch_size_X_;
		size_t EM_patch_size_Y_;
		size_t EM_numEventMatching_;
		size_t EM_patch_intensity_threshold_;
		double EM_patch_valid_ratio_;

		// Event Block Matching (BM) parameters
		double BM_half_slice_thickness_;
		size_t BM_MAX_NUM_EVENTS_PER_MATCHING_;
		size_t BM_patch_size_X_;
		size_t BM_patch_size_Y_;
		size_t BM_min_disparity_;
		size_t BM_max_disparity_;
		size_t BM_step_;
		double BM_ZNCC_Threshold_;
		bool BM_bUpDownConfiguration_;

		// SGM [45] parameters
		int num_disparities_;
		int block_size_;
		int P1_;
		int P2_;
		int uniqueness_ratio_;
		cv::Ptr<cv::StereoSGBM> sgbm_;

		ros::Publisher pose_pub_;
		image_transport::Publisher invDepthMap_pub_, stdVarMap_pub_, ageMap_pub_, costMap_pub_;
		// For counting the total number of fusion
		size_t TotalNumFusion_;

		std::string resultPath_;

		// EMVS_Mapping
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
		// pcl::PointCloud<pcl::PointXYZI>::Ptr emvs_pc_, pc_map_;

		double meanDepth_;
		double KEYFRAME_LINEAR_DIS_, KEYFRAME_ORIENTATION_DIS_, KEYFRAME_MEANDEPTH_DIS_;
		image_transport::Publisher depthMap_pub_, confidenceMap_pub_, semiDenseMask_pub_, varianceMap_pub_;
		int EMVS_Keyframe_event_, EMVS_Init_event_;

		bool SAVE_RESULT_;
		std::string strDataset_;

		std::map<ros::Time, std::shared_ptr<Eigen::MatrixXd>> TS_map_negative_history_;
		Eigen::Matrix4d T_world_map_;
	};

} // namespace esvo_core

#endif //ESVO_CORE_ESVO_MVSMONO_H