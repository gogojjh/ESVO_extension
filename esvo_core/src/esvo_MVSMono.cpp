#include <esvo_core/esvo_MVSMono.h>
#include <esvo_core/DVS_MappingStereoConfig.h>
#include <esvo_core/tools/params_helper.h>

#include <minkindr_conversions/kindr_tf.h>
#include <geometry_msgs/TransformStamped.h>

#include <opencv2/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>

#include <thread>
#include <iterator>
#include <memory>
#include <algorithm>
#include <utility>

//#define ESVO_MVSMONO_LOG

namespace esvo_core
{
	esvo_MVSMono::esvo_MVSMono(
		const ros::NodeHandle &nh,
		const ros::NodeHandle &nh_private)
		: nh_(nh),
		  pnh_(nh_private),
		  it_(nh),
		  calibInfoDir_(tools::param(pnh_, "calibInfoDir", std::string(""))),
		  camSysPtr_(new CameraSystem(calibInfoDir_, false)),
		  dpConfigPtr_(new DepthProblemConfig(
			  tools::param(pnh_, "patch_size_X", 25),
			  tools::param(pnh_, "patch_size_Y", 25),
			  tools::param(pnh_, "LSnorm", std::string("Tdist")),
			  tools::param(pnh_, "Tdist_nu", 0.0),
			  tools::param(pnh_, "Tdist_scale", 0.0),
			  tools::param(pnh_, "ITERATION_OPTIMIZATION", 10))),
		  invDepth_min_range_(tools::param(pnh_, "invDepth_min_range", 0.16)),
		  invDepth_max_range_(tools::param(pnh_, "invDepth_max_range", 2.0)),
		  pc_(new PointCloud()),
		  pc_global_(new PointCloud()),
		  depthFramePtr_(new DepthFrame(camSysPtr_->cam_left_ptr_->height_, camSysPtr_->cam_left_ptr_->width_)),
		  // MapperEMVS
		  emvs_dsi_shape_(tools::param(pnh_, "opts_dim_x", 0),
						  tools::param(pnh_, "opts_dim_y", 0),
						  tools::param(pnh_, "opts_dim_z", 100),
						  1.0 / invDepth_max_range_, 1.0 / invDepth_min_range_, 0.0),
		  emvs_opts_depth_map_(tools::param(pnh_, "opts_depth_map_kernal_size", 5),
							   tools::param(pnh_, "opts_depth_map_threshold_c", 5),
							   tools::param(pnh_, "opts_depth_map_median_filter_size", 5),
							   tools::param(pnh_, "opts_depth_map_contrast_threshold", 70)),
		  emvs_opts_pc_(tools::param(pnh_, "opts_pc_radius_search", 0.05),
						tools::param(pnh_, "opts_pc_min_num_neighbors", 3)),
		  emvs_opts_mapper_(tools::param(pnh_, "opts_mapper_parallex", 3.0),
							tools::param(pnh_, "opts_mapper_PatchSize_X", 5),
							tools::param(pnh_, "opts_mapper_PatchSize_Y", 5),
							tools::param(pnh_, "opts_mapper_TS_score", 50.0)),
		  emvs_mapper_(camSysPtr_->cam_left_ptr_, emvs_dsi_shape_, emvs_opts_mapper_)
	{
		// frame id
		dvs_frame_id_ = tools::param(pnh_, "dvs_frame_id", std::string("dvs"));
		world_frame_id_ = tools::param(pnh_, "world_frame_id", std::string("world"));
		pc_->header.frame_id = world_frame_id_;
		pc_global_->header.frame_id = world_frame_id_;

		/**** online parameters ***/
		// options
		bDenoising_ = tools::param(pnh_, "Denoising", false);
		resetButton_ = tools::param(pnh_, "ResetButton", false);

		// module parameters
		PROCESS_EVENT_NUM_ = tools::param(pnh_, "PROCESS_EVENT_NUM", 500);
		TS_HISTORY_LENGTH_ = tools::param(pnh_, "TS_HISTORY_LENGTH", 100);
		mapping_rate_hz_ = tools::param(pnh_, "mapping_rate_hz", 20);

		// callbacks functions
		events_left_sub_ = nh_.subscribe<dvs_msgs::EventArray>("events_left", 10, boost::bind(&esvo_MVSMono::eventsCallback, this, _1, boost::ref(events_left_)));
		stampedPose_sub_ = nh_.subscribe("stamped_pose", 10, &esvo_MVSMono::stampedPoseCallback, this);
		TS_left_sub_ = nh_.subscribe("time_surface_left", 10, &esvo_MVSMono::timeSurfaceCallback, this);
		// TF
		tf_ = std::make_shared<tf::Transformer>(true, ros::Duration(100.0));

		// result publishers
		pc_pub_ = nh_.advertise<PointCloud>("/esvo_mvsmono/pointcloud_local", 1);
		pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/esvo_mvsmono/pose_pub", 1);
		if (bVisualizeGlobalPC_)
		{
			gpc_pub_ = nh_.advertise<PointCloud>("/esvo_mvsmono/pointcloud_global", 1);
			pc_global_->reserve(500000);
			t_last_pub_pc_ = 0.0;
		}
		bVisualizeGlobalPC_ = tools::param(pnh_, "bVisualizeGlobalPC", false);
		visualizeGPC_interval_ = tools::param(pnh_, "visualizeGPC_interval", 3);
		numAddedPC_threshold_ = tools::param(pnh_, "NumGPC_added_oper_refresh", 1000);

		resultPath_ = tools::param(pnh_, "PATH_TO_SAVE_RESULT", std::string());
		strDataset_ = tools::param(pnh_, "Dataset_Name", std::string());

		KEYFRAME_MEANDEPTH_DIS_ = tools::param(pnh_, "KEYFRAME_MEANDEPTH_DIS", 0.15);  // percentage

		// DSI_Mapper configure
		emvs_dsi_shape_.printDSIInfo();

		isKeyframe_ = true;
		T_w_frame_.setIdentity();
		T_w_keyframe_.setIdentity();
		meanDepth_ = -1.0;
		mAllPoses_.clear();

		depthMap_pub_ = it_.advertise("DSI_Depth_Map", 1);
		confidenceMap_pub_ = it_.advertise("DSI_Confidence_Map", 1);
		semiDenseMask_pub_ = it_.advertise("DSI_Semi_Dense_Mask", 1);
		varianceMap_pub_ = it_.advertise("DSI_Variance_Map", 1);
		EMVS_Init_event_ = tools::param(pnh_, "EMVS_Init_event", 5e4);
		EMVS_Keyframe_event_ = tools::param(pnh_, "EMVS_Keyframe_event", 2e5); // it should be: EMVS_Keyframe_event_ > EMVS_Init_event_
		SAVE_RESULT_ = tools::param(pnh_, "SAVE_RESULT", false);
		T_world_map_.setIdentity();

		// multi-thread management
		mapping_thread_future_ = mapping_thread_promise_.get_future();
		reset_future_ = reset_promise_.get_future();

		// stereo mapping detached thread
		std::thread MappingThread(&esvo_MVSMono::MappingLoop, this,
								  std::move(mapping_thread_promise_), std::move(reset_future_));
		MappingThread.detach();

		// Dynamic reconfigure
		// dynamic_reconfigure_callback_ = boost::bind(&esvo_MVSMono::onlineParameterChangeCallback, this, _1, _2);
		// server_.reset(new dynamic_reconfigure::Server<DVS_MappingStereoConfig>(nh_private));
		// server_->setCallback(dynamic_reconfigure_callback_);

		solverFlag_ = INITIAL;
	}

	esvo_MVSMono::~esvo_MVSMono()
	{
		pc_pub_.shutdown();
		pose_pub_.shutdown();
	}

	void esvo_MVSMono::MappingLoop(
		std::promise<void> prom_mapping,
		std::future<void> future_reset)
	{
		ros::Rate r(mapping_rate_hz_);

		while (ros::ok())
		{
			if (changed_frame_rate_)
			{
				printf("Changing mapping framerate to %lu Hz", mapping_rate_hz_);
				r = ros::Rate(mapping_rate_hz_);
				changed_frame_rate_ = false;
			}
			//
			if (TS_history_.size() >= 10) /* To assure the esvo_time_surface node has been working. */
			{
#ifdef ESVO_MVSMONO_LOG
				LOG(INFO) << "TS_history_.size(): " << TS_history_.size();
#endif
				while (true)
				{
					if (data_mutex_.try_lock())
					{
						dataTransferring();
						data_mutex_.unlock();
						break;
					}
					else
					{
						if (future_reset.wait_for(std::chrono::nanoseconds(1)) == std::future_status::ready)
						{
							prom_mapping.set_value();
							return;
						}
					}
				}

				// To check if the most current TS observation has been loaded by dataTransferring()
				if (TS_obs_.second.isEmpty())
				{
					r.sleep();
					continue;
				}
				publishKFPose(TS_obs_.first, TS_obs_.second.T_w_obs_);
				MappingAtTime(TS_obs_.first);
				insertKeyframe();
			}
			else
			{
				if (future_reset.wait_for(std::chrono::nanoseconds(1)) == std::future_status::ready)
				{
					prom_mapping.set_value();
					return;
				}
			}
			r.sleep();
		}
	}

	void esvo_MVSMono::MappingAtTime(const ros::Time &t)
	{
		TicToc tt_mapping;
		double t_overall_count = 0;

		/******************************************************************/
		/*************** Event Multi-Stereo Mapping (EMVS) ****************/
		/******************************************************************/
		double t_denoising;
		if (bDenoising_) // Set it to "True" to deal with flicker effect caused by VICON.
		{
			tt_mapping.tic();
			// Draw one mask image for denoising
			cv::Mat denoising_mask;
			createDenoisingMask(vALLEventsPtr_left_, denoising_mask, camSysPtr_->cam_left_ptr_->height_, camSysPtr_->cam_left_ptr_->width_);

			// Extract events (appear on edges likely) for each TS
			vDenoisedEventsPtr_left_.clear();
			extractDenoisedEvents(vCloseEventsPtr_left_, vDenoisedEventsPtr_left_, denoising_mask, PROCESS_EVENT_NUM_);
			totalNumCount_ = vDenoisedEventsPtr_left_.size();
			t_denoising = tt_mapping.toc();
		}
		else
		{
			vDenoisedEventsPtr_left_.clear();
			vDenoisedEventsPtr_left_.reserve(PROCESS_EVENT_NUM_);
			vDenoisedEventsPtr_left_.insert(vDenoisedEventsPtr_left_.end(), vCloseEventsPtr_left_.begin(),
											vCloseEventsPtr_left_.begin() + min(vCloseEventsPtr_left_.size(), PROCESS_EVENT_NUM_));
			// vDenoisedEventsPtr_left_.insert(vDenoisedEventsPtr_left_.end(), vCloseEventsPtr_left_.begin(), vCloseEventsPtr_left_.end());
		}

		// undistort events' coordinates
		std::vector<Eigen::Vector4d> vEdgeletCoordinates;
		vEdgeletCoordinates.reserve(vDenoisedEventsPtr_left_.size());
		for (size_t i = 0; i < vDenoisedEventsPtr_left_.size(); i++)
		{
			// undistortion + rectification
			bool bDistorted = true;
			Eigen::Vector2d coor;
			if (bDistorted)
				coor = camSysPtr_->cam_left_ptr_->getRectifiedUndistortedCoordinate(vDenoisedEventsPtr_left_[i]->x, vDenoisedEventsPtr_left_[i]->y);
			else
				coor = Eigen::Vector2d(vDenoisedEventsPtr_left_[i]->x, vDenoisedEventsPtr_left_[i]->y);
			Eigen::Vector4d tmp_coor;
			tmp_coor[0] = coor[0];
			tmp_coor[1] = coor[1];
			tmp_coor[2] = vDenoisedEventsPtr_left_[i]->ts.toSec();
			tmp_coor[3] = double(vDenoisedEventsPtr_left_[i]->polarity);
			vEdgeletCoordinates.push_back(tmp_coor);
		}

		double t_solve;
		tt_mapping.tic();
		if (isKeyframe_)
		{
			LOG(INFO) << "insert a keyframe: reset the DSI for the local map";
			if (SAVE_RESULT_ && boost::filesystem::exists(resultPath_ + strDataset_))
				emvs_mapper_.dsi_.writeGridNpy(std::string(resultPath_ + strDataset_ + "dsi.npy").c_str());
			emvs_mapper_.reset();
			emvs_mapper_.initializeDSI(TS_obs_.second.T_w_obs_);
			meanDepth_ = 0.0;
		}
		else
		{
			// LOG(INFO) << "insert an non-keyframe: add events onto the DSI";
		}

		emvs_mapper_.storeEventsPose(mVirtualPoses_, vEdgeletCoordinates);
		emvs_mapper_.updateDSI();
		emvs_mapper_.clearEvents();
		cv::Mat depth_map, confidence_map, semidense_mask;
		emvs_mapper_.getDepthMapFromDSI(depth_map, confidence_map, semidense_mask, emvs_opts_depth_map_, meanDepth_);
		t_solve = tt_mapping.toc();
		LOG_EVERY_N(INFO, 50) << "Time to update DSI and count DSI: " << t_solve << "ms"; // 40ms
		LOG_EVERY_N(INFO, 50) << "Mean square = " << emvs_mapper_.dsi_.computeMeanSquare();
		// visualization
		std::thread tPublishDSIResult(&esvo_MVSMono::publishDSIResults, this,
										t, semidense_mask, depth_map, confidence_map);
		tPublishDSIResult.detach();
		LOG_EVERY_N(INFO, 50) << "Accumulate events: " << emvs_mapper_.accu_event_number_;
		if (emvs_mapper_.accu_event_number_ >= EMVS_Keyframe_event_)
		{
			std::vector<DepthPoint> vdp;
			emvs_mapper_.getDepthPoint(depth_map, confidence_map, semidense_mask, vdp, 0.1);
			propagatePoints(vdp, emvs_mapper_.T_w_rv_, pc_);
			LOG_EVERY_N(INFO, 20) << "Depth point size: " << vdp.size() << ", process events: " << emvs_mapper_.accu_event_number_;
		}

		std::thread tPublishMappingResult(&esvo_MVSMono::publishEMVSPointCloud, this, t);
		tPublishMappingResult.detach();

#ifdef ESVO_MVSMONO_LOG
		LOG(INFO) << "\n";
		LOG(INFO) << "------------------------------------------------------------";
		LOG(INFO) << "--------------------Computation Cost-------------------------";
		LOG(INFO) << "------------------------------------------------------------";
		LOG(INFO) << "Denoising: " << t_denoising << " ms, (" << t_denoising / t_overall_count * 100 << "%).";
		LOG(INFO) << "------------------------------------------------------------";
		LOG(INFO) << "-- nonlinear optimization: " << t_solve << " ms, (" << t_solve / t_overall_count * 100
				  << "%).";
		LOG(INFO) << "------------------------------------------------------------";
		LOG(INFO) << "------------------------------END---------------------------";
		LOG(INFO) << "------------------------------------------------------------";
		LOG(INFO) << "\n";
#endif
		return;
	}

	/**
	 * @brief: This function defines xx criterias to indicate if insert new keyframe in Mapping
	 */
	void esvo_MVSMono::insertKeyframe()
	{
		T_w_frame_ = TS_obs_.second.T_w_obs_;
		double dis = (T_w_frame_.topRightCorner<3, 1>() - T_w_keyframe_.topRightCorner<3, 1>()).norm();
		if (meanDepth_ > 0.0 && dis > KEYFRAME_MEANDEPTH_DIS_ * meanDepth_) // 1e5: system just starts; > 0.15: move at a long distance
		{
			isKeyframe_ = true;
			T_w_keyframe_ = T_w_frame_;
			// LOG(INFO) << "mean depth: " << meanDepth_ << "m";
			meanDepth_ = -1.0;
		}
		else
		{
			isKeyframe_ = false;
		}
	}

	void esvo_MVSMono::propagatePoints(const std::vector<DepthPoint> &vdp,
									   const Eigen::Matrix4d &T_world_frame,
									   PointCloud::Ptr &pc_ptr)
	{
		pc_ptr->clear();
		pc_ptr->reserve(vdp.size());
		for (size_t i = 0; i < vdp.size(); i++)
		{
			Eigen::Vector3d p_prop = T_world_frame.topLeftCorner<3, 3>() * vdp[i].p_cam() + T_world_frame.topRightCorner<3, 1>();
			pcl::PointXYZ point;
			point.x = p_prop.x();
			point.y = p_prop.y();
			point.z = p_prop.z();
			pc_ptr->push_back(point);
		}
	}

	bool esvo_MVSMono::dataTransferring()
	{
		TS_obs_ = std::make_pair(ros::Time(), TimeSurfaceObservation()); // clean the TS obs.
		if (TS_history_.size() <= 10)
			return false;
		totalNumCount_ = 0;

		// load current Time-Surface Observation
		auto it_end = TS_history_.rbegin();
		it_end++; // in case that the tf is behind the most current TS.
		auto it_begin = TS_history_.begin();
		while (TS_obs_.second.isEmpty())
		{
			Eigen::Matrix4d T_w_obs;
			if (trajectory_.getPoseAt(mAllPoses_, it_end->first, T_w_obs)) // check if poses are ready
			{
				it_end->second.setTransformation(T_w_obs);
				TS_obs_ = *it_end;
			}
			if (it_end->first == it_begin->first)
				break;
			it_end++;
		}
		if (TS_obs_.second.isEmpty())
			return false;

		// copy all involved events' pointers
		vALLEventsPtr_left_.clear();   // to generate denoising mask (only used for ECCV 2018 dataset)
		vCloseEventsPtr_left_.clear(); // will be denoised using the mask above.

		// load allEvent
		double dt_events = 1.0 / mapping_rate_hz_; // 0.05s
		double dt_pose = 0.00025;
		ros::Time t_end = TS_obs_.first;
		ros::Time t_begin(std::max(0.0, t_end.toSec() - dt_events));
		auto ev_begin_it = tools::EventBuffer_upper_bound(events_left_, t_begin); //events_left_.begin();
		auto ev_end_it = tools::EventBuffer_lower_bound(events_left_, t_end) - 1;
		size_t MAX_NUM_Event_INVOLVED;
		if (solverFlag_ == INITIAL)
			MAX_NUM_Event_INVOLVED = 1e6;
		else
			MAX_NUM_Event_INVOLVED = 1e5;
		// const size_t MAX_NUM_Event_INVOLVED = static_cast<size_t>(ev_end_it - ev_begin_it);
		vALLEventsPtr_left_.reserve(MAX_NUM_Event_INVOLVED);
		vCloseEventsPtr_left_.reserve(MAX_NUM_Event_INVOLVED);
		while (ev_end_it != ev_begin_it && vALLEventsPtr_left_.size() < MAX_NUM_Event_INVOLVED)
		{
			vALLEventsPtr_left_.push_back(ev_end_it._M_cur);
			vCloseEventsPtr_left_.push_back(ev_end_it._M_cur);
			ev_end_it--;
		}
		totalNumCount_ = vCloseEventsPtr_left_.size();

		// using inherent linear interpolation
		// t0 <= ev.ts <= t1
		//   pose_0 (t0), pose_1 (t1), ..., pose_N (tN)
		// t_begin, t_tmp, ...,                     t_end
		mVirtualPoses_.clear();
		mVirtualPoses_.reserve(dt_events / dt_pose + 1);
		ros::Time t_tmp = t_end;
		while (t_tmp.toSec() > t_begin.toSec() - dt_pose)
		{
			Eigen::Matrix4d T;
			if (trajectory_.getPoseAt(mAllPoses_, t_tmp, T))
			{
				mVirtualPoses_.push_back(std::make_pair(t_tmp, T));
			}
			else
			{
				LOG(INFO) << "waiting for new poses for events";
				return false;
			}
			t_tmp = ros::Time(t_tmp.toSec() - dt_pose);
		}
		return true;
	}

	void esvo_MVSMono::stampedPoseCallback(const geometry_msgs::PoseStampedConstPtr &ps_msg)
	{
		std::lock_guard<std::mutex> lock(data_mutex_);
		// To check inconsistent timestamps and reset.
		static constexpr double max_time_diff_before_reset_s = 0.5;
		const ros::Time stamp_first_event = ps_msg->header.stamp;
		std::string *err_tf = new std::string();
		int iGetLastest_common_time =
			tf_->getLatestCommonTime(dvs_frame_id_.c_str(), world_frame_id_.c_str(), tf_lastest_common_time_, err_tf);
		delete err_tf;

		if (tf_lastest_common_time_.toSec() != 0)
		{
			const double dt = stamp_first_event.toSec() - tf_lastest_common_time_.toSec();
			if (dt < 0 || std::fabs(dt) >= max_time_diff_before_reset_s)
			{
				printf("Inconsistent event timestamps detected <stampedPoseCallback> (new: %f, old %f), resetting.",
					   stamp_first_event.toSec(), tf_lastest_common_time_.toSec());
				reset();
			}
		}

		// compatitable to esvo_Tracking
		// obtain T_world_cam
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
		Eigen::Matrix4d T_marker_cam;
		if (!strDataset_.compare("rpg_mono"))
		{
			T_marker_cam.setIdentity();
		}
		else if (!strDataset_.compare("rpg_stereo"))
		{
			T_marker_cam << 5.363262328777285e-01, -1.748374625145743e-02, -8.438296573030597e-01, -7.009849865398374e-02,
				8.433577587813513e-01, -2.821937531845164e-02, 5.366109927684415e-01, 1.881333563905305e-02,
				-3.319431623758162e-02, -9.994488408486204e-01, -3.897382049768972e-04, -6.966829200678797e-02,
				0, 0, 0, 1;
		}
		else if (!strDataset_.compare("rpg_slider"))
		{
			T_marker_cam.setIdentity();
		}
		else if (!strDataset_.compare("rpg_simu"))
		{
			T_marker_cam.setIdentity();
		}
		else if (!strDataset_.compare("upenn"))
		{
			T_marker_cam.setIdentity();
		}
		else if (!strDataset_.compare("ust_mono"))
		{
			T_marker_cam.setIdentity();
		}
		else if (!strDataset_.compare("ust_stereo"))
		{
			T_marker_cam.setIdentity();
		}
		else
		{
			T_marker_cam.setIdentity();
		}

		Eigen::Matrix4d T_world_cam = T_world_marker * T_marker_cam;
		if (T_world_map_ == Eigen::Matrix4d::Identity())
			T_world_map_ = T_world_cam;
		Eigen::Matrix4d T_map_cam = T_world_map_.inverse() * T_world_cam;

		Eigen::Matrix3d R_map_cam = T_map_cam.topLeftCorner<3, 3>();
		Eigen::Quaterniond q_map_cam(R_map_cam);
		Eigen::Vector3d t_map_cam = T_map_cam.topRightCorner<3, 1>();

		// add pose to tf
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
		tf::StampedTransform st(tf, ps_msg->header.stamp, world_frame_id_.c_str(), dvs_frame_id_.c_str());
		tf_->setTransform(st);

		// static tf::TransformBroadcaster br;
		// br.sendTransform(st);
		mAllPoses_.emplace(ps_msg->header.stamp, T_map_cam);
		static constexpr size_t MAX_POSE_LENGTH = 1000; // the buffer size
		if (mAllPoses_.size() > 2000)
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

	void esvo_MVSMono::eventsCallback(const dvs_msgs::EventArray::ConstPtr &msg, EventQueue &EQ)
	{
		std::lock_guard<std::mutex> lock(data_mutex_);

		static constexpr double max_time_diff_before_reset_s = 0.5;
		const ros::Time stamp_first_event = msg->events[0].ts;

		// check time stamp inconsistency
		if (!msg->events.empty() && !EQ.empty())
		{
			const double dt = stamp_first_event.toSec() - EQ.back().ts.toSec();
			if (dt < 0 || std::fabs(dt) >= max_time_diff_before_reset_s)
			{
				printf("Inconsistent event timestamps detected <eventCallback> (new: %f, old %f), resetting.",
						 stamp_first_event.toSec(), events_left_.back().ts.toSec());
				reset();
			}
		}

		// add new ones and remove old ones
		for (const dvs_msgs::Event &e : msg->events)
		{
			EQ.push_back(e);
			int i = EQ.size() - 2;
			while (i >= 0 && EQ[i].ts > e.ts) // we may have to sort the queue, just in case the raw event messages do not come in a chronological order.
			{
				EQ[i + 1] = EQ[i];
				i--;
			}
			EQ[i + 1] = e;
		}
		clearEventQueue(EQ);
	}

	void esvo_MVSMono::clearEventQueue(EventQueue &EQ)
	{
		static constexpr size_t MAX_EVENT_QUEUE_LENGTH = 3000000;
		if (EQ.size() > MAX_EVENT_QUEUE_LENGTH)
		{
			size_t NUM_EVENTS_TO_REMOVE = EQ.size() - MAX_EVENT_QUEUE_LENGTH;
			EQ.erase(EQ.begin(), EQ.begin() + NUM_EVENTS_TO_REMOVE);
		}
	}

	void esvo_MVSMono::timeSurfaceCallback(const sensor_msgs::ImageConstPtr &time_surface_left)
	{
		std::lock_guard<std::mutex> lock(data_mutex_);
		// check time-stamp inconsistency
		if (!TS_history_.empty())
		{
			static constexpr double max_time_diff_before_reset_s = 0.5;
			const ros::Time stamp_last_image = TS_history_.rbegin()->first;
			const double dt = time_surface_left->header.stamp.toSec() - stamp_last_image.toSec();
			if (dt < 0 || std::fabs(dt) >= max_time_diff_before_reset_s)
			{
				printf("Inconsistent frame timestamp detected <timeSurfaceCallback> (new: %f, old %f), resetting.",
						 time_surface_left->header.stamp.toSec(), stamp_last_image.toSec());
				reset();
			}
		}

		cv_bridge::CvImagePtr cv_ptr_left, cv_ptr_right;
		try
		{
			cv_ptr_left = cv_bridge::toCvCopy(time_surface_left, sensor_msgs::image_encodings::MONO8);
			cv_ptr_right = cv_bridge::toCvCopy(time_surface_left, sensor_msgs::image_encodings::MONO8);
			// cv_ptr_right = cv_bridge::toCvCopy(time_surface_right, sensor_msgs::image_encodings::MONO8);
		}
		catch (cv_bridge::Exception &e)
		{
			ROS_ERROR("cv_bridge exception: %s", e.what());
			return;
		}

		// push back the new time surface map
		ros::Time t_new_TS = time_surface_left->header.stamp;
		// Made the gradient computation optional which is up to the jacobian choice.
		TS_history_.emplace(t_new_TS, TimeSurfaceObservation(cv_ptr_left, cv_ptr_right, TS_id_));
		TS_id_++;

		// keep TS_history's size constant
		while (TS_history_.size() > TS_HISTORY_LENGTH_)
		{
			auto it = TS_history_.begin();
			TS_history_.erase(it);
		}
	}

	void esvo_MVSMono::reset()
	{
		// mutual-thread communication with MappingThread.
		LOG(INFO) << "Coming into reset()";
		reset_promise_.set_value();
		LOG(INFO) << "(reset) The mapping thread future is waiting for the value.";
		mapping_thread_future_.get();
		LOG(INFO) << "(reset) The mapping thread future receives the value.";

		// clear all maintained data
		events_left_.clear();
		TS_history_.clear();
		tf_->clear();
		pc_->clear();
		pc_global_->clear();
		TS_id_ = 0;
		depthFramePtr_->clear();
		dqvDepthPoints_.clear();
		LOG(INFO) << "****************************************************";
		LOG(INFO) << "****************** RESET THE SYSTEM *********************";
		LOG(INFO) << "****************************************************\n";

		// restart the mapping thread
		reset_promise_ = std::promise<void>();
		mapping_thread_promise_ = std::promise<void>();
		reset_future_ = reset_promise_.get_future();
		mapping_thread_future_ = mapping_thread_promise_.get_future();
		std::thread MappingThread(&esvo_MVSMono::MappingLoop, this,
								  std::move(mapping_thread_promise_), std::move(reset_future_));
		MappingThread.detach();

		T_world_map_.setIdentity();
	}

	void esvo_MVSMono::onlineParameterChangeCallback(DVS_MappingStereoConfig &config, uint32_t level)
	{
		bool have_display_parameters_changed = false;
		{
			std::lock_guard<std::mutex> lock(data_mutex_);

			if (invDepth_min_range_ != config.invDepth_min_range ||
				invDepth_max_range_ != config.invDepth_max_range ||
				bDenoising_ != config.Denoising ||
				bRegularization_ != config.Regularization ||
				resetButton_ != config.ResetButton ||
				PROCESS_EVENT_NUM_ != config.PROCESS_EVENT_NUM ||
				TS_HISTORY_LENGTH_ != config.TS_HISTORY_LENGTH)
			{
				have_display_parameters_changed = true;
			}

			invDepth_min_range_ = config.invDepth_min_range;
			invDepth_max_range_ = config.invDepth_max_range;
			bDenoising_ = config.Denoising;
			resetButton_ = config.ResetButton;
			PROCESS_EVENT_NUM_ = config.PROCESS_EVENT_NUM;
			TS_HISTORY_LENGTH_ = config.TS_HISTORY_LENGTH;
		}

		if (config.mapping_rate_hz != mapping_rate_hz_)
		{
			changed_frame_rate_ = true;
			have_display_parameters_changed = true;
			mapping_rate_hz_ = config.mapping_rate_hz;
		}

		if (have_display_parameters_changed)
		{
			std::lock_guard<std::mutex> lock(data_mutex_);
			LOG(INFO) << "onlineParameterChangeCallback ==============";
			reset();
		}
	}

	void esvo_MVSMono::publishDSIResults(const ros::Time &t, const cv::Mat &semiDenseMask,
										 const cv::Mat &depthMap, const cv::Mat &confidenceMap)
	{
		publishImage(255 * semiDenseMask, t, semiDenseMask_pub_, "mono8");

		cv::Mat depthMap255 = (depthMap - emvs_dsi_shape_.min_depth_) * (255.0 / (emvs_dsi_shape_.max_depth_ - emvs_dsi_shape_.min_depth_));
		cv::Mat depthMap8Bit, depthMapColor;
		depthMap255.convertTo(depthMap8Bit, CV_8U);
		cv::applyColorMap(depthMap8Bit, depthMapColor, cv::COLORMAP_RAINBOW);
		cv::Mat depthOnCanvas = cv::Mat(depthMap.rows, depthMap.cols, CV_8UC3, cv::Scalar(1, 1, 1) * 255);
		depthMapColor.copyTo(depthOnCanvas, semiDenseMask);
		publishImage(depthMapColor, t, depthMap_pub_, "bgr8");

		cv::Mat confidenceMap255;
		cv::normalize(confidenceMap, confidenceMap255, 0, 255.0, cv::NORM_MINMAX, CV_8UC1);
		publishImage(confidenceMap255, t, confidenceMap_pub_, "mono8");
	}

	void esvo_MVSMono::publishEMVSPointCloud(const ros::Time &t)
	{
		sensor_msgs::PointCloud2::Ptr pc_to_publish(new sensor_msgs::PointCloud2);
		if (!pc_->empty())
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::RadiusOutlierRemoval<pcl::PointXYZ> outlier_rm;
			outlier_rm.setInputCloud(pc_);
			outlier_rm.setRadiusSearch(emvs_opts_pc_.radius_search_);
			outlier_rm.setMinNeighborsInRadius(emvs_opts_pc_.min_num_neighbors_);
			outlier_rm.filter(*cloud_filtered);
			pc_->swap(*cloud_filtered);

			pcl::toROSMsg(*pc_, *pc_to_publish);
			pc_to_publish->header.stamp = t;
			pc_pub_.publish(pc_to_publish);

			// publish global pointcloud
			if (bVisualizeGlobalPC_)
			{
				if (t.toSec() - t_last_pub_pc_ > visualizeGPC_interval_)
				{
					PointCloud::Ptr pc_filtered(new PointCloud());
					pcl::VoxelGrid<pcl::PointXYZ> sor;
					sor.setInputCloud(pc_);
					sor.setLeafSize(0.03, 0.03, 0.03);
					sor.filter(*pc_filtered);

					// copy the most current pc tp pc_global
					size_t pc_length = pc_filtered->size();
					if (pc_length == 0)
						return;
					size_t numAddedPC = min(pc_length, numAddedPC_threshold_) - 1;
					pc_global_->insert(pc_global_->end(), pc_filtered->end() - numAddedPC, pc_filtered->end());

					// publish point cloud
					pcl::toROSMsg(*pc_global_, *pc_to_publish);
					pc_to_publish->header.stamp = t;
					gpc_pub_.publish(pc_to_publish);
					t_last_pub_pc_ = t.toSec();
				}
			}
		}
	}

	void esvo_MVSMono::publishKFPose(const ros::Time &t, const Eigen::Matrix4d &T)
	{
		geometry_msgs::PoseStampedPtr ps_ptr(new geometry_msgs::PoseStamped());
		Eigen::Vector3d tr = T.topRightCorner<3, 1>();
		Eigen::Matrix3d R = T.topLeftCorner<3, 3>();
		Eigen::Quaterniond q(R);
		ps_ptr->header.stamp = t;
		ps_ptr->header.frame_id = world_frame_id_;
		ps_ptr->pose.position.x = tr(0);
		ps_ptr->pose.position.y = tr(1);
		ps_ptr->pose.position.z = tr(2);
		ps_ptr->pose.orientation.x = q.x();
		ps_ptr->pose.orientation.y = q.y();
		ps_ptr->pose.orientation.z = q.z();
		ps_ptr->pose.orientation.w = q.w();
		pose_pub_.publish(ps_ptr);
	}

	void esvo_MVSMono::publishImage(const cv::Mat &image,
									const ros::Time &t,
									image_transport::Publisher &pub,
									std::string encoding)
	{
		if (pub.getNumSubscribers() == 0)
			return;

		std_msgs::Header header;
		header.stamp = t;
		sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, encoding.c_str(), image).toImageMsg();
		pub.publish(msg);
	}

	void esvo_MVSMono::createEdgeMask(std::vector<dvs_msgs::Event *> &vEventsPtr,
									  PerspectiveCamera::Ptr &camPtr,
									  cv::Mat &edgeMap,
									  std::vector<std::pair<size_t, size_t>> &vEdgeletCoordinates,
									  bool bUndistortEvents,
									  size_t radius)
	{
		size_t col = camPtr->width_;
		size_t row = camPtr->height_;
		int dilate_radius = (int)radius;
		edgeMap = cv::Mat(cv::Size(col, row), CV_8UC1, cv::Scalar(0));
		vEdgeletCoordinates.reserve(col * row);

		auto it_tmp = vEventsPtr.begin();
		while (it_tmp != vEventsPtr.end())
		{
			// undistortion + rectification
			Eigen::Matrix<double, 2, 1> coor;
			if (bUndistortEvents)
				coor = camPtr->getRectifiedUndistortedCoordinate((*it_tmp)->x, (*it_tmp)->y);
			else
				coor = Eigen::Matrix<double, 2, 1>((*it_tmp)->x, (*it_tmp)->y);

			// assign
			int xcoor = std::floor(coor(0));
			int ycoor = std::floor(coor(1));

			for (int dy = -dilate_radius; dy <= dilate_radius; dy++)
				for (int dx = -dilate_radius; dx <= dilate_radius; dx++)
				{
					int x = xcoor + dx;
					int y = ycoor + dy;
					if (x < 0 || x >= col || y < 0 || y >= row)
						continue;
					else
					{
						edgeMap.at<uchar>(y, x) = 255;
						vEdgeletCoordinates.emplace_back((size_t)x, (size_t)y);
					}
				}
			it_tmp++;
		}
	}

	void esvo_MVSMono::createDenoisingMask(
		std::vector<dvs_msgs::Event *> &vAllEventsPtr,
		cv::Mat &mask,
		size_t row, size_t col)
	{
		cv::Mat eventMap;
		visualizor_.plot_eventMap(vAllEventsPtr, eventMap, row, col);
		cv::medianBlur(eventMap, mask, 3);
	}

	void esvo_MVSMono::extractDenoisedEvents(
		std::vector<dvs_msgs::Event *> &vCloseEventsPtr,
		std::vector<dvs_msgs::Event *> &vEdgeEventsPtr,
		cv::Mat &mask,
		size_t maxNum)
	{
		vEdgeEventsPtr.reserve(vCloseEventsPtr.size());
		for (size_t i = 0; i < vCloseEventsPtr.size(); i++)
		{
			if (vEdgeEventsPtr.size() >= maxNum)
				break;
			size_t x = vCloseEventsPtr[i]->x;
			size_t y = vCloseEventsPtr[i]->y;
			if (mask.at<uchar>(y, x) == 255)
				vEdgeEventsPtr.push_back(vCloseEventsPtr[i]);
		}
	}

} // namespace esvo_core