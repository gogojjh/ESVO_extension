#include <esvo_core/Tracking.h>
#include <esvo_core/tools/TicToc.h>
#include <esvo_core/tools/params_helper.h>
#include <minkindr_conversions/kindr_tf.h>
#include <tf/transform_broadcaster.h>
#include <sys/stat.h>

namespace esvo_core
{
	Tracking::Tracking(
		const ros::NodeHandle &nh,
		const ros::NodeHandle &nh_private) : nh_(nh),
											 pnh_(nh_private),
											 it_(nh),
											 calibInfoDir_(tools::param(pnh_, "calibInfoDir", std::string(""))),
											 camSysPtr_(new CameraSystem(calibInfoDir_, false)),
											 rpConfigPtr_(new RegProblemConfig(
												 tools::param(pnh_, "patch_size_X", 25),
												 tools::param(pnh_, "patch_size_Y", 25),
												 tools::param(pnh_, "kernelSize", 15),
												 tools::param(pnh_, "LSnorm", std::string("l2")),
												 tools::param(pnh_, "huber_threshold", 10.0),
												 tools::param(pnh_, "invDepth_min_range", 0.0),
												 tools::param(pnh_, "invDepth_max_range", 0.0),
												 tools::param(pnh_, "MIN_NUM_EVENTS", 1000),
												 tools::param(pnh_, "MAX_REGISTRATION_POINTS", 500),
												 tools::param(pnh_, "BATCH_SIZE", 200),
												 tools::param(pnh_, "MAX_ITERATION", 10))),
											 rpType_((RegProblemType)((size_t)tools::param(pnh_, "RegProblemType", 0))),
											 rpSolver_(camSysPtr_, rpConfigPtr_, rpType_, NUM_THREAD_TRACKING),
											 ESVO_System_Status_("INITIALIZATION"),
											 ets_(IDLE)
	{
		// offline data
		dvs_frame_id_ = tools::param(pnh_, "dvs_frame_id", std::string("dvs"));
		world_frame_id_ = tools::param(pnh_, "world_frame_id", std::string("world"));

		/**** online parameters ***/
		nh_.setParam("/ESVO_SYSTEM_STATUS", ESVO_System_Status_);
		tracking_rate_hz_ = tools::param(pnh_, "tracking_rate_hz", 100);
		TS_HISTORY_LENGTH_ = tools::param(pnh_, "TS_HISTORY_LENGTH", 100);
		REF_HISTORY_LENGTH_ = tools::param(pnh_, "REF_HISTORY_LENGTH", 5);
		bSaveTrajectory_ = tools::param(pnh_, "SAVE_TRAJECTORY", false);
		bVisualizeTrajectory_ = tools::param(pnh_, "VISUALIZE_TRAJECTORY", true);
		resultPath_ = tools::param(pnh_, "PATH_TO_SAVE_TRAJECTORY", std::string());
		strDataset_ = tools::param(pnh_, "Dataset_Name", std::string("rpg"));
		strSequence_ = tools::param(pnh_, "Sequence_Name", std::string("shapes_poster"));
		strRep_ = tools::param(pnh_, "Representation_Name", std::string("TS"));

		// online data callbacks
		events_left_sub_ = nh_.subscribe("events_left", 0, &Tracking::eventsCallback, this);
		TS_left_sub_ = nh_.subscribe("time_surface_left", 10, &Tracking::timeSurfaceCallback, this);
		map_sub_ = nh_.subscribe("pointcloud", 0, &Tracking::refMapCallback, this);				// local map in the ref view.
		stampedPose_sub_ = nh_.subscribe("stamped_pose", 0, &Tracking::stampedPoseCallback, this); // for accessing the pose of the ref view.
		gtPose_sub_ = nh_.subscribe("gt_pose", 0, &Tracking::gtPoseCallback, this); // for accessing the pose of the ref view.
		// TF
		tf_ = std::make_shared<tf::Transformer>(true, ros::Duration(100.0));

		pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/tracking/pose_pub", 1);
		path_pub_ = nh_.advertise<nav_msgs::Path>("/tracking/trajectory", 1);
		pose_gt_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/gt/pose_pub", 1);
		path_gt_pub_ = nh_.advertise<nav_msgs::Path>("/gt/trajectory", 1);

		/*** For Visualization and Test ***/
		reprojMap_pub_left_ = it_.advertise("Reproj_Map_Left", 1);
		rpSolver_.setRegPublisher(&reprojMap_pub_left_);
		ref_.tr_.setIdentity();

		/*** Tracker ***/
		if (!strRep_.compare("TS"))
		{
			std::thread TrackingThread(&Tracking::TrackingLoopTS, this);
			TrackingThread.detach();
		} 
		else if (!strRep_.compare("EM"))
		{
			std::thread TrackingThread(&Tracking::TrackingLoopEM, this);
			TrackingThread.detach();
		}
		else
		{
			LOG(INFO) << "Please select proper event frame-based Representation!";
		}

		T_world_cur_.setIdentity();
		T_world_map_.setIdentity();
		last_gt_timestamp_ = 0.0;
		last_save_trajectory_timestamp_ = 0.0;

		num_NewEvents_ = 0;
	}

	Tracking::~Tracking()
	{
		pose_pub_.shutdown();
	}

	void Tracking::TrackingLoopTS()
	{
		ros::Rate r(tracking_rate_hz_);
		while (true)
		{
			if (!ros::ok())
				break;
			nh_.getParam("/ESVO_SYSTEM_STATUS", ESVO_System_Status_);

			if (ESVO_System_Status_ == "TERMINATE")
			{
				LOG(INFO) << "The tracking node is terminated manually...";
				break;
			}

			if (ESVO_System_Status_ == "RESET") // This is true when the system is mannally reset, waiting for mapping's response
			{
				r.sleep();
				continue;
			}

			if (ESVO_System_Status_ == "INITIALIZATION" && ets_ == WORKING) // This is true when the system is reset from dynamic reconfigure
			{
				reset();
				r.sleep();
				continue;
			}

			if (!TS_buf_.empty())
			{
				m_buf_.lock();
				if (cur_.t_.toSec() < TS_buf_.back().first.toSec()) // new observation arrived
				{
					if (ref_.t_.toSec() >= TS_buf_.back().first.toSec())
					{
						LOG(INFO) << "The time_surface observation should be obtained after the reference frame";
						exit(-1);
					}
					curDataTransferring(); // set current data
				}
				else
				{
					m_buf_.unlock();
					r.sleep();
					continue;
				}

				if (refPCMap_buf_.empty()) // Tracking and Mapping are still in initialization
				{
					// publishTimeSurface(cur_.t_);
					publishPose(cur_.t_, cur_.tr_); // publish identity pose
					if (bVisualizeTrajectory_)
						publishPath(cur_.t_, cur_.tr_);
					m_buf_.unlock();
					r.sleep();
					continue;
				}
				else if (ref_.t_.toSec() < refPCMap_buf_.back().first.toSec()) // new reference map arrived
				{
					refDataTransferring(); // set reference data
					// LOG(INFO) << "receive map from Mapping";
				}	
				m_buf_.unlock();
			
				// create new regProblem
				TicToc tt;
				double t_resetRegProblem, t_solve, t_pub_result;
				if (rpSolver_.resetRegProblem(&ref_, &cur_)) // will be false if no enough points in local map, need to reinitialize
				{
					t_resetRegProblem = tt.toc();
					if (ets_ == IDLE)
						ets_ = WORKING;
					if (ESVO_System_Status_ != "WORKING")
						nh_.setParam("/ESVO_SYSTEM_STATUS", "WORKING"); // trigger the main mapping process

					tt.tic();
					if (rpType_ == REG_NUMERICAL)
						rpSolver_.solve_numerical();
					if (rpType_ == REG_ANALYTICAL) // default: analytical
						rpSolver_.solve_analytical();
					t_solve = tt.toc();
					T_world_cur_ = cur_.tr_.getTransformationMatrix();

					tt.tic();
					publishPose(cur_.t_, cur_.tr_);
					if (bVisualizeTrajectory_)
						publishPath(cur_.t_, cur_.tr_);
					t_pub_result = tt.toc();

					// save result and gt if available.
					if (bSaveTrajectory_)
					{
						// save results to listPose and listPoseGt
						lTimestamp_.push_back(std::to_string(cur_.t_.toSec()));
						lPose_.push_back(cur_.tr_.getTransformationMatrix());
					}
					m_buf_.lock();
					// publishTimeSurface(cur_.t_);
					m_buf_.unlock();
				}
				else
				{
					nh_.setParam("/ESVO_SYSTEM_STATUS", "INITIALIZATION");
					ets_ = IDLE;
					LOG(INFO) << "Tracking thread is IDLE";
					m_buf_.lock();
					// publishTimeSurface(cur_.t_);
					m_buf_.unlock();
				}

				double t_overall_count = 0;
				t_overall_count = t_resetRegProblem + t_solve + t_pub_result;
#ifdef ESVO_CORE_TRACKING_LOG
				LOG(INFO) << "\n";
				LOG(INFO) << "------------------------------------------------------------";
				LOG(INFO) << "--------------------Tracking Computation Cost (TS)----------";
				LOG(INFO) << "------------------------------------------------------------";
				LOG(INFO) << "ResetRegProblem: " << t_resetRegProblem << " ms, (" << t_resetRegProblem / t_overall_count * 100 << "%).";
				LOG(INFO) << "Registration: " << t_solve << " ms, (" << t_solve / t_overall_count * 100 << "%).";
				LOG(INFO) << "pub result: " << t_pub_result << " ms, (" << t_pub_result / t_overall_count * 100 << "%).";
				LOG(INFO) << "Total Computation (" << rpSolver_.lmStatics_.nPoints_ << "): " << t_overall_count << " ms.";
				LOG(INFO) << "------------------------------------------------------------";
				LOG(INFO) << "------------------------------------------------------------";
#endif
				if (bSaveTrajectory_ && (cur_.t_.toSec() - last_save_trajectory_timestamp_ > 1.0))
				{
					last_save_trajectory_timestamp_ = cur_.t_.toSec();
					struct stat st;
					if (stat(resultPath_.c_str(), &st) == -1) // there is no such dir, create one
					{
						LOG(INFO) << "There is no such directory: " << resultPath_;
						_mkdir(resultPath_.c_str());
						LOG(INFO) << "The directory has been created!!!";
					}
#ifdef ESVO_CORE_TRACKING_LOG
					// LOG(INFO) << "pose size: " << lPose_.size() << ", refPCMap_buf size(): " << refPCMap_buf_.size() << ", TS_buf.size(): " << TS_buf_.size();
#endif
					saveTrajectory(resultPath_ + strDataset_ + "/" + strSequence_ + "_" + strRep_ + "_traj_estimate.txt", lTimestamp_, lPose_);
					saveTrajectory(resultPath_ + strDataset_ + "/" + strSequence_ + "_" + strRep_ + "_traj_gt.txt", lTimestamp_GT_, lPose_GT_);
					std::unordered_map<std::string, double> umTimeCost;
					umTimeCost["resetReg"] = t_resetRegProblem;
					umTimeCost["solveReg"] = t_solve;
					umTimeCost["pubReg"] = t_pub_result;
					umTimeCost["total"] = t_overall_count;
					vTimeCost_.push_back(umTimeCost);
					saveTimeCost(resultPath_ + strDataset_ + "/" + strSequence_ + "_" + strRep_ + "_time.txt", vTimeCost_);
				}
			}
			r.sleep();
		} // while
	}

	void Tracking::TrackingLoopEM()
	{
		ros::Rate r(tracking_rate_hz_);
		while (true)
		{
			if (!ros::ok())
				break;
			nh_.getParam("/ESVO_SYSTEM_STATUS", ESVO_System_Status_);

			if (ESVO_System_Status_ == "TERMINATE")
			{
				LOG(INFO) << "The tracking node is terminated manually...";
				break;
			}

			if (ESVO_System_Status_ == "RESET") // This is true when the system is mannally reset, waiting for mapping's response
			{
				r.sleep();
				continue;
			}

			if (ESVO_System_Status_ == "INITIALIZATION" && ets_ == WORKING) // This is true when the system is reset from dynamic reconfigure
			{
				reset();
				r.sleep();
				continue;
			}

			const size_t MAX_NUM_Event_INVOLVED = 2000;
			if (num_NewEvents_ >= MAX_NUM_Event_INVOLVED && events_left_.size() > MAX_NUM_Event_INVOLVED)
			{
				m_buf_.lock();
				TicToc t_pre;
				double t_LoadEvent, t_CurDataTransfer, t_RefDataTransfer;

				std::vector<dvs_msgs::Event *> vEventSubsetPtr;
				vEventSubsetPtr.reserve(MAX_NUM_Event_INVOLVED);
				auto ev_begin_it = events_left_.end() - MAX_NUM_Event_INVOLVED;
				auto ev_end_it = events_left_.end();
				while (ev_begin_it != ev_end_it)
				{
					vEventSubsetPtr.push_back(ev_begin_it._M_cur);
					ev_begin_it++;
				}

				if (ref_.t_.toSec() >= vEventSubsetPtr.back()->ts.toSec())
				{
					LOG(INFO) << "The time_surface observation should be obtained after the reference frame";
					exit(-1);
				}
				t_LoadEvent = t_pre.toc();
				t_pre.tic();

				size_t col = camSysPtr_->cam_left_ptr_->width_;
				size_t row = camSysPtr_->cam_left_ptr_->height_;
				cv::Mat eventMap = cv::Mat(cv::Size(col, row), CV_8UC1, cv::Scalar(0));
				for (size_t i = 0; i < vEventSubsetPtr.size(); i++)
					eventMap.at<uchar>(vEventSubsetPtr[i]->y, vEventSubsetPtr[i]->x) = 255;

				// curDataTransferring
				cur_.t_ = ros::Time((vEventSubsetPtr.front()->ts.toSec() + vEventSubsetPtr.back()->ts.toSec()) / 2);
				std_msgs::Header header;
				header.stamp = ros::Time(cur_.t_);
				header.frame_id = world_frame_id_;
				cv_bridge::CvImagePtr cv_ptr_left(new cv_bridge::CvImage(header, "mono8", eventMap));
				cv_bridge::CvImagePtr cv_ptr_right(new cv_bridge::CvImage(header, "mono8", eventMap));
				TimeSurfaceObservation TS_obs_fake(cv_ptr_left, cv_ptr_right, 0, false);
				cur_.pTsObs_ = &TS_obs_fake;
				cur_.tr_ = Transformation(T_world_cur_);
				cur_.numEventsSinceLastObs_ = vEventSubsetPtr.size();
				t_CurDataTransfer = t_pre.toc();

				// cur_.pTsObs_->getTimeSurfaceNegative(5);
				// cv::Mat TS_negative_mat;
				// cv::eigen2cv(cur_.pTsObs_->TS_negative_left_, TS_negative_mat);
				// TS_negative_mat.convertTo(TS_negative_mat, CV_8UC1);
				// cv::imshow("negative TS", TS_negative_mat);
				// cv::waitKey(15);

				if (refPCMap_buf_.empty()) // Tracking and Mapping are still in initialization
				{
					// publishTimeSurface(cur_.t_);
					publishPose(cur_.t_, cur_.tr_); // publish identity pose
					if (bVisualizeTrajectory_)
						publishPath(cur_.t_, cur_.tr_);
					m_buf_.unlock();
					r.sleep();
					continue;
				}
				else if (ref_.t_.toSec() < refPCMap_buf_.back().first.toSec()) // new reference map arrived
				{
					refDataTransferring(); // set reference data
										   // LOG(INFO) << "receive map from Mapping";
				}
				m_buf_.unlock();

				// create new regProblem
				TicToc tt;
				double t_resetRegProblem, t_solve, t_pub_result;
				if (rpSolver_.resetRegProblem(&ref_, &cur_)) // will be false if no enough points in local map, need to reinitialize
				{
					t_resetRegProblem = tt.toc();
					if (ets_ == IDLE)
						ets_ = WORKING;
					if (ESVO_System_Status_ != "WORKING")
						nh_.setParam("/ESVO_SYSTEM_STATUS", "WORKING"); // trigger the main mapping process

					tt.tic();
					if (rpType_ == REG_NUMERICAL)
						rpSolver_.solve_numerical();
					if (rpType_ == REG_ANALYTICAL) // default: analytical
						rpSolver_.solve_analytical();
					t_solve = tt.toc();
					T_world_cur_ = cur_.tr_.getTransformationMatrix();

					tt.tic();
					publishPose(cur_.t_, cur_.tr_);
					if (bVisualizeTrajectory_)
						publishPath(cur_.t_, cur_.tr_);
					t_pub_result = tt.toc();

					// save result and gt if available.
					if (bSaveTrajectory_)
					{
						// save results to listPose and listPoseGt
						lTimestamp_.push_back(std::to_string(cur_.t_.toSec()));
						lPose_.push_back(cur_.tr_.getTransformationMatrix());
					}
					m_buf_.lock();
					// publishTimeSurface(cur_.t_);
					m_buf_.unlock();
				}
				else
				{
					nh_.setParam("/ESVO_SYSTEM_STATUS", "INITIALIZATION");
					ets_ = IDLE;
					LOG(INFO) << "Tracking thread is IDLE";
					m_buf_.lock();
					// publishTimeSurface(cur_.t_);
					m_buf_.unlock();
				}

				double t_overall_count = 0;
				t_overall_count = t_resetRegProblem + t_solve + t_pub_result;
#ifdef ESVO_CORE_TRACKING_LOG
				LOG(INFO) << "\n";
				LOG(INFO) << "------------------------------------------------------------";
				LOG(INFO) << "--------------------Tracking Computation Cost (EM)----------";
				LOG(INFO) << "------------------------------------------------------------";
				// LOG(INFO) << "Load events: " << t_LoadEvent << " ms"; // 0.00791ms
				// LOG(INFO) << "Current data transfer: " << t_CurDataTransfer << " ms"; // 0.45ms
				LOG(INFO) << "ResetRegProblem: " << t_resetRegProblem << " ms, (" << t_resetRegProblem / t_overall_count * 100 << "%).";
				LOG(INFO) << "Registration: " << t_solve << " ms, (" << t_solve / t_overall_count * 100 << "%).";
				LOG(INFO) << "pub result: " << t_pub_result << " ms, (" << t_pub_result / t_overall_count * 100 << "%).";
				LOG(INFO) << "Total Computation (" << rpSolver_.lmStatics_.nPoints_ << "): " << t_overall_count << " ms.";
				LOG(INFO) << "------------------------------------------------------------";
				LOG(INFO) << "------------------------------------------------------------";
#endif
				if (bSaveTrajectory_ && (cur_.t_.toSec() - last_save_trajectory_timestamp_ > 1.0))
				{
					last_save_trajectory_timestamp_ = cur_.t_.toSec();
					struct stat st;
					if (stat(resultPath_.c_str(), &st) == -1) // there is no such dir, create one
					{
						LOG(INFO) << "There is no such directory: " << resultPath_;
						_mkdir(resultPath_.c_str());
						LOG(INFO) << "The directory has been created!!!";
					}
#ifdef ESVO_CORE_TRACKING_LOG
					// LOG(INFO) << "pose size: " << lPose_.size() << ", refPCMap_buf size(): " << refPCMap_buf_.size() << ", TS_buf.size(): " << TS_buf_.size();
#endif
					saveTrajectory(resultPath_ + strDataset_ + "/" + strSequence_ + "_" + strRep_ + "_traj_estimate.txt", lTimestamp_, lPose_);
					saveTrajectory(resultPath_ + strDataset_ + "/" + strSequence_ + "_" + strRep_ + "_traj_gt.txt", lTimestamp_GT_, lPose_GT_);
					std::unordered_map<std::string, double> umTimeCost;
					umTimeCost["resetReg"] = t_resetRegProblem;
					umTimeCost["solveReg"] = t_solve;
					umTimeCost["pubReg"] = t_pub_result;
					umTimeCost["total"] = t_overall_count;
					vTimeCost_.push_back(umTimeCost);
					saveTimeCost(resultPath_ + strDataset_ + "/" + strSequence_ + "_" + strRep_ + "_time.txt", vTimeCost_);
				}
				num_NewEvents_ = 0;
			}
			r.sleep();
		} // while
	}

	/**
    * @brief reload the current point cloud
    **/
	bool Tracking::refDataTransferring()
	{
		// load reference info
		ref_.t_ = refPCMap_buf_.back().first;
		if (ESVO_System_Status_ == "INITIALIZATION" && ets_ == IDLE) // will be true if receive the first PCMap from mapping
			ref_.tr_.setIdentity();
		if (ESVO_System_Status_ == "WORKING" || (ESVO_System_Status_ == "INITIALIZATION" && ets_ == WORKING))
		{
			// if (!getPoseAt(ref_.t_, ref_.tr_, dvs_frame_id_))
			// {
			// 	LOG(INFO) << "ESVO_System_Status_: " << ESVO_System_Status_ << ", ref_.t_: " << ref_.t_.toNSec();
			// 	LOG(INFO) << "Logic error ! There must be a pose for the given timestamp, because mapping has been finished.";
			// 	exit(-1);
			// 	return false;
			// }
			// std::cout << ref_.tr_.getTransformationMatrix() << std::endl;
			ref_.tr_ = cur_.tr_;
		}
		size_t numPoint = refPCMap_buf_.back().second->size();
		ref_.vPointXYZPtr_.clear();
		ref_.vPointXYZPtr_.reserve(numPoint);
		auto PointXYZ_begin_it = refPCMap_buf_.back().second->begin();
		auto PointXYZ_end_it = refPCMap_buf_.back().second->end();
		while (PointXYZ_begin_it != PointXYZ_end_it)
		{
			ref_.vPointXYZPtr_.push_back(PointXYZ_begin_it.base()); // Copy the pointer of the pointXYZ
			PointXYZ_begin_it++;
		}
		return true;
	}

	/**
    * @brief extract current events
    **/
	bool Tracking::curDataTransferring()
	{
		// load current observation
		auto ev_begin_it = EventBuffer_lower_bound(events_left_, cur_.t_);
		cur_.t_ = TS_buf_.back().first;
		cur_.pTsObs_ = &TS_buf_.back().second;
		cur_.tr_ = Transformation(T_world_cur_);
		auto ev_end_it = EventBuffer_lower_bound(events_left_, cur_.t_);
		cur_.numEventsSinceLastObs_ = std::distance(ev_begin_it, ev_end_it) + 1; // Count the number of events occuring since the last observation.
		// LOG(INFO) << "event number in 10ms: " << cur_.numEventsSinceLastObs_; // 2000-1400
		return true;
	}

	void Tracking::reset()
	{
		m_buf_.lock();
		// clear all maintained data
		ets_ = IDLE;
		TS_id_ = 0;
		TS_buf_.clear();
		refPCMap_buf_.clear();
		events_left_.clear();

		path_.poses.clear();
		path_gt_.poses.clear();
		m_buf_.unlock();
	}

	/********************** Callback functions *****************************/
	void Tracking::refMapCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
	{
		m_buf_.lock();
		PointCloud::Ptr PC_ptr(new PointCloud());
		pcl::fromROSMsg(*msg, *PC_ptr);
		refPCMap_buf_.emplace_back(msg->header.stamp, PC_ptr);
		while (refPCMap_buf_.size() > REF_HISTORY_LENGTH_) // 10
			refPCMap_buf_.pop_front();
		m_buf_.unlock();
	}

	void Tracking::eventsCallback(const dvs_msgs::EventArray::ConstPtr &msg)
	{
		// std::lock_guard<std::mutex> lock(data_mutex_);
		m_buf_.lock();
		// add new ones and remove old ones
		for (const dvs_msgs::Event &e : msg->events)
		{
			events_left_.push_back(e);
			int i = events_left_.size() - 2;
			while (i >= 0 && events_left_[i].ts > e.ts) // we may have to sort the queue, just in case the raw event messages do not come in a chronological order.
			{
				events_left_[i + 1] = events_left_[i];
				i--;
			}
			events_left_[i + 1] = e;
		}
		clearEventQueue();

		num_NewEvents_ += msg->events.size();
		m_buf_.unlock();
	}

	void Tracking::clearEventQueue()
	{
		static constexpr size_t MAX_EVENT_QUEUE_LENGTH = 5000000;
		if (events_left_.size() > MAX_EVENT_QUEUE_LENGTH)
		{
			static constexpr size_t MAX_EVENT_QUEUE_LENGTH = 5000000;
			if (events_left_.size() > MAX_EVENT_QUEUE_LENGTH)
			{
				size_t remove_events = events_left_.size() - MAX_EVENT_QUEUE_LENGTH;
				events_left_.erase(events_left_.begin(), events_left_.begin() + remove_events);
			}
		}
	}

	void Tracking::timeSurfaceCallback(const sensor_msgs::ImageConstPtr &time_surface_left)
	{
		// std::lock_guard<std::mutex> lock(data_mutex_);
		cv_bridge::CvImagePtr cv_ptr_left, cv_ptr_right;
		try
		{
			cv_ptr_left = cv_bridge::toCvCopy(time_surface_left, sensor_msgs::image_encodings::MONO8);
			cv_ptr_right = cv_bridge::toCvCopy(time_surface_left, sensor_msgs::image_encodings::MONO8);
		}
		catch (cv_bridge::Exception &e)
		{
			ROS_ERROR("cv_bridge exception: %s", e.what());
			return;
		}

		m_buf_.lock();
		ros::Time t_new_ts = time_surface_left->header.stamp;
		TS_buf_.push_back(std::make_pair(t_new_ts, TimeSurfaceObservation(cv_ptr_left, cv_ptr_right, TS_id_, false)));
		TS_id_++;
		while (TS_buf_.size() > TS_HISTORY_LENGTH_)
			TS_buf_.pop_front();
		m_buf_.unlock();
	}

	void Tracking::stampedPoseCallback(const geometry_msgs::PoseStampedConstPtr &ps_msg)
	{
		m_buf_.lock();
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
		m_buf_.unlock();
	}

	void Tracking::gtPoseCallback(const geometry_msgs::PoseStampedConstPtr &ps_msg)
	{
		// m_buf_.lock();
		if (ps_msg->header.stamp.toSec() - last_gt_timestamp_ > 0.01)
		{
			last_gt_timestamp_ = ps_msg->header.stamp.toSec();
		}
		else
		{
			return;
		}
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
			// T_marker_cam << 1.0, 0.0, 0.0, 0.0,
			// 				0.0, -1.0, 0.0, 0.0,
			// 				0.0, 0.0, -1.0, 0.0,
			// 				0.0, 0.0, 0.0, 1.0;
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

		// save gt pose
		if (bSaveTrajectory_)
		{
			lPose_GT_.push_back(T_map_cam);
			lTimestamp_GT_.push_back(std::to_string(ps_msg->header.stamp.toSec()));
		}
		// m_buf_.unlock();
	}

	bool Tracking::getPoseAt(const ros::Time &t, esvo_core::Transformation &Tr,
							 const std::string &source_frame)
	{
		std::string *err_msg = new std::string();
		if (!tf_->canTransform(world_frame_id_, source_frame, t, err_msg))
		{
			LOG(WARNING) << t << " : " << *err_msg;
			delete err_msg;
			return false;
		}
		else
		{
			tf::StampedTransform st;
			tf_->lookupTransform(world_frame_id_, source_frame, t, st);
			tf::transformTFToKindr(st, &Tr);
			return true;
		}
	}

	void Tracking::publishPose(const ros::Time &t, Transformation &tr)
	{
		geometry_msgs::PoseStampedPtr ps_ptr(new geometry_msgs::PoseStamped());
		ps_ptr->header.stamp = t;
		ps_ptr->header.frame_id = world_frame_id_;
		ps_ptr->pose.position.x = tr.getPosition()(0);
		ps_ptr->pose.position.y = tr.getPosition()(1);
		ps_ptr->pose.position.z = tr.getPosition()(2);
		ps_ptr->pose.orientation.x = tr.getRotation().x();
		ps_ptr->pose.orientation.y = tr.getRotation().y();
		ps_ptr->pose.orientation.z = tr.getRotation().z();
		ps_ptr->pose.orientation.w = tr.getRotation().w();
		pose_pub_.publish(ps_ptr);
	}

	void Tracking::publishPath(const ros::Time &t, Transformation &tr)
	{
		geometry_msgs::PoseStampedPtr ps_ptr(new geometry_msgs::PoseStamped());
		ps_ptr->header.stamp = t;
		ps_ptr->header.frame_id = world_frame_id_;
		ps_ptr->pose.position.x = tr.getPosition()(0);
		ps_ptr->pose.position.y = tr.getPosition()(1);
		ps_ptr->pose.position.z = tr.getPosition()(2);
		ps_ptr->pose.orientation.x = tr.getRotation().x();
		ps_ptr->pose.orientation.y = tr.getRotation().y();
		ps_ptr->pose.orientation.z = tr.getRotation().z();
		ps_ptr->pose.orientation.w = tr.getRotation().w();

		path_.header.stamp = t;
		path_.header.frame_id = world_frame_id_;
		path_.poses.push_back(*ps_ptr);
		path_pub_.publish(path_);
	}

	void Tracking::saveTrajectory(const std::string &resultDir,
								  const std::list<std::string> &lTimestamp,
								  const std::list<Eigen::Matrix<double, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 4>>> &lPose)

	{
#ifdef ESVO_CORE_TRACKING_DEBUG
		LOG(INFO) << "Saving trajectory to " << resultDir << " ......";
#endif
		std::ofstream f;
		f.open(resultDir.c_str(), std::ofstream::out);
		if (!f.is_open())
		{
			LOG(INFO) << "File at " << resultDir << " is not opened, save trajectory failed.";
			exit(-1);
		}
		f << std::fixed;

		std::list<Eigen::Matrix<double, 4, 4>,
				  Eigen::aligned_allocator<Eigen::Matrix<double, 4, 4>>>::const_iterator result_it_begin = lPose.begin();
		std::list<Eigen::Matrix<double, 4, 4>,
				  Eigen::aligned_allocator<Eigen::Matrix<double, 4, 4>>>::const_iterator result_it_end = lPose.end();
		std::list<std::string>::const_iterator ts_it_begin = lTimestamp.begin();

		for (; result_it_begin != result_it_end; result_it_begin++, ts_it_begin++)
		{
			Eigen::Matrix3d Rwc_result;
			Eigen::Vector3d twc_result;
			Rwc_result = (*result_it_begin).block<3, 3>(0, 0);
			twc_result = (*result_it_begin).block<3, 1>(0, 3);
			Eigen::Quaterniond q(Rwc_result);
			f << *ts_it_begin << " " << std::setprecision(9)
			  << twc_result.x() << " " << twc_result.y() << " " << twc_result.z() << " "
			  << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
		}
		f.close();
#ifdef ESVO_CORE_TRACKING_DEBUG
		LOG(INFO) << "Saving trajectory to " << resultDir << ". Done !!!!!!.";
#endif
	}

	void Tracking::saveTimeCost(const std::string &resultDir,
								const std::vector<std::unordered_map<std::string, double>> &vTimeCost_)
	{
		std::ofstream f;
		f.open(resultDir.c_str(), std::ofstream::out);
		if (!f.is_open())
		{
			LOG(INFO) << "File at " << resultDir << " is not opened, save trajectory failed.";
			exit(-1);
		}
		f << std::fixed;
		for (auto it_TimeCost = vTimeCost_.begin(); it_TimeCost != vTimeCost_.end(); it_TimeCost++)
		{
			for (auto it = it_TimeCost->begin(); it != it_TimeCost->end(); it++)
			{
				f << it->second << " ";
			}
			f << std::endl;
		}
		f.close();
	}

} // namespace esvo_core