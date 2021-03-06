#ifndef ESVO_CORE_ESVO_MVSMONO_H
#define ESVO_CORE_ESVO_MVSMONO_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <tf/transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include <esvo_core/container/DepthPoint.h>
#include <esvo_core/container/CameraSystem.h>
#include <esvo_core/container/TimeSurfaceObservation.h>
#include <esvo_core/core/DepthProblemConfig.h>
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

// #define ESVO_MVSMONO_LOG

namespace esvo_core
{
	using namespace core;
	enum SolverFlag
	{
		INITIAL,
		MAPPING
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
		bool InitializationAtTime(const ros::Time &t);
		void MappingAtTime(const ros::Time &t);
		bool dataTransferring();
		void propagatePoints(const std::vector<DepthPoint> &vdp,
							 const Eigen::Matrix4d &T_world_frame,
							 PointCloud::Ptr &pc_ptr);

		// callback functions
		void stampedPoseCallback(const geometry_msgs::PoseStampedConstPtr &ps_msg);
		void eventsCallback(const dvs_msgs::EventArray::ConstPtr &msg, EventQueue &EQ);
		void timeSurfaceCallback(const sensor_msgs::ImageConstPtr &time_surface_left);
		void onlineParameterChangeCallback(DVS_MappingStereoConfig &config, uint32_t level);

		// utils
		void clearEventQueue(EventQueue &EQ);
		void reset();

		// results
		void publishEMVSPointCloud(
			const ros::Time &t);
		void publishImage(
			const cv::Mat &image,
			const ros::Time &t,
			image_transport::Publisher &pub,
			std::string encoding = "bgr8");
		void publishKFPose(const ros::Time &t, const Eigen::Matrix4d &T);

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
		ros::Publisher pc_pub_, gpc_pub_, emvs_pc_pub_;
		image_transport::ImageTransport it_;

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
		std::shared_ptr<tf::Transformer> tf_;
		size_t TS_id_;
		ros::Time tf_lastest_common_time_;

		// system
		DepthProblemConfig::Ptr dpConfigPtr_;
		Visualization visualizor_;

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
		// module parameters
		size_t PROCESS_EVENT_NUM_;
		size_t TS_HISTORY_LENGTH_;
		size_t mapping_rate_hz_;
		// options
		bool changed_frame_rate_;
		bool bRegularization_;
		bool resetButton_;
		bool bDenoising_;

		ros::Publisher pose_pub_;

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
		Eigen::Matrix4d T_w_keyframe_, T_w_frame_, T_world_map_;

		double meanDepth_;
		double KEYFRAME_LINEAR_DIS_, KEYFRAME_ORIENTATION_DIS_, KEYFRAME_MEANDEPTH_DIS_;
		image_transport::Publisher depthMap_pub_, confidenceMap_pub_, semiDenseMask_pub_, varianceMap_pub_;
		int EMVS_Keyframe_event_, EMVS_Init_event_;

		bool SAVE_RESULT_;
		std::string strDataset_;
		SolverFlag solverFlag_;
	};

} // namespace esvo_core

#endif //ESVO_CORE_ESVO_MVSMONO_H