#include <esvo_core/esvo_Tracking.h>
#include <esvo_core/tools/TicToc.h>
#include <esvo_core/tools/params_helper.h>
#include <minkindr_conversions/kindr_tf.h>
#include <tf/transform_broadcaster.h>
#include <sys/stat.h>

//#define ESVO_CORE_TRACKING_DEBUG
//#define ESVO_CORE_TRACKING_DEBUG

namespace esvo_core
{
	esvo_Tracking::esvo_Tracking(
		const ros::NodeHandle &nh,
		const ros::NodeHandle &nh_private) : nh_(nh),
											 pnh_(nh_private),
											 it_(nh),
											 TS_left_sub_(nh_, "time_surface_left", 10),
											 TS_right_sub_(nh_, "time_surface_right", 10),
											 TS_sync_(ExactSyncPolicy(10), TS_left_sub_, TS_right_sub_),
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
		tracking_rate_hz_ = tools::param(pnh_, "tracking_rate_hz", 100);
		TS_HISTORY_LENGTH_ = tools::param(pnh_, "TS_HISTORY_LENGTH", 100);
		REF_HISTORY_LENGTH_ = tools::param(pnh_, "REF_HISTORY_LENGTH", 5);
		bSaveTrajectory_ = tools::param(pnh_, "SAVE_TRAJECTORY", false);
		bVisualizeTrajectory_ = tools::param(pnh_, "VISUALIZE_TRAJECTORY", true);
		resultPath_ = tools::param(pnh_, "PATH_TO_SAVE_TRAJECTORY", std::string());
		nh_.setParam("/ESVO_SYSTEM_STATUS", ESVO_System_Status_);

		// online data callbacks
		events_left_sub_ = nh_.subscribe<dvs_msgs::EventArray>(
			"events_left", 0, &esvo_Tracking::eventsCallback, this);
		TS_sync_.registerCallback(boost::bind(&esvo_Tracking::timeSurfaceCallback, this, _1, _2));
		tf_ = std::make_shared<tf::Transformer>(true, ros::Duration(100.0));
		map_sub_ = nh_.subscribe("pointcloud", 0, &esvo_Tracking::refMapCallback, this);				// local map in the ref view.
		stampedPose_sub_ = nh_.subscribe("stamped_pose", 0, &esvo_Tracking::stampedPoseCallback, this); // for accessing the pose of the ref view.
		gtPose_sub_ = nh_.subscribe("gt_pose", 0, &esvo_Tracking::gtPoseCallback, this); // for accessing the pose of the ref view.

		pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/esvo_tracking/pose_pub", 1);
		path_pub_ = nh_.advertise<nav_msgs::Path>("/esvo_tracking/trajectory", 1);
		pose_gt_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/gt/pose_pub", 1);
		path_gt_pub_ = nh_.advertise<nav_msgs::Path>("/gt/trajectory", 1);

		/*** For Visualization and Test ***/
		reprojMap_pub_left_ = it_.advertise("Reproj_Map_Left", 1);
		rpSolver_.setRegPublisher(&reprojMap_pub_left_);

		/*** Tracker ***/
		T_world_cur_ = Eigen::Matrix<double, 4, 4>::Identity();
		std::thread TrackingThread(&esvo_Tracking::TrackingLoop, this);
		TrackingThread.detach();

		time_surface_left_pub_ = it_.advertise("/esvo_tracking/TS_left", 1);
		time_surface_right_pub_ = it_.advertise("/esvo_tracking/TS_right", 1);
	}

	esvo_Tracking::~esvo_Tracking()
	{
		pose_pub_.shutdown();
	}

	void esvo_Tracking::TrackingLoop()
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
					curDataTransferring();
				}
				else
				{
					m_buf_.unlock();
					r.sleep();
					continue;
				}

				if (refPCMap_buf_.empty()) // Mapping is still in initialization
				{
					publishTimeSurface();
					publishPose(cur_.t_, cur_.tr_); // publish identity pose
					if (bVisualizeTrajectory_)
						publishPath(cur_.t_, cur_.tr_);
					m_buf_.unlock();
					r.sleep();
					continue;
				}
				else if (ref_.t_.toSec() < refPCMap_buf_.back().first.toSec()) // new reference map arrived
				{
					refDataTransferring();
				}	
				m_buf_.unlock();
			
				// create new regProblem
				TicToc tt;
				double t_resetRegProblem, t_solve, t_pub_result, t_pub_gt;
				if (rpSolver_.resetRegProblem(&ref_, &cur_)) // will be false if no enough points in local map, need to reinitialize
				{
					if (ets_ == IDLE)
						ets_ = WORKING;
					if (ESVO_System_Status_ != "WORKING")
						nh_.setParam("/ESVO_SYSTEM_STATUS", "WORKING"); // trigger the main mapping process

					if (rpType_ == REG_NUMERICAL)
						rpSolver_.solve_numerical();
					if (rpType_ == REG_ANALYTICAL)
						rpSolver_.solve_analytical();

					T_world_cur_ = cur_.tr_.getTransformationMatrix();
					publishPose(cur_.t_, cur_.tr_);
					if (bVisualizeTrajectory_)
						publishPath(cur_.t_, cur_.tr_);

					// save result and gt if available.
					if (bSaveTrajectory_)
					{
						// save results to listPose and listPoseGt
						lTimestamp_.push_back(std::to_string(cur_.t_.toSec()));
						lPose_.push_back(cur_.tr_.getTransformationMatrix());
					}
					m_buf_.lock();
					publishTimeSurface();
					m_buf_.unlock();
				}
				else
				{
					nh_.setParam("/ESVO_SYSTEM_STATUS", "INITIALIZATION");
					ets_ = IDLE;
					LOG(INFO) << "Tracking thread is IDLE";
					m_buf_.lock();
					publishTimeSurface();
					m_buf_.unlock();
				}

#ifdef ESVO_CORE_TRACKING_LOG
				double t_overall_count = 0;
				t_overall_count = t_resetRegProblem + t_solve + t_pub_result;
				LOG(INFO) << "\n";
				LOG(INFO) << "------------------------------------------------------------";
				LOG(INFO) << "--------------------Tracking Computation Cost---------------";
				LOG(INFO) << "------------------------------------------------------------";
				LOG(INFO) << "ResetRegProblem: " << t_resetRegProblem << " ms, (" << t_resetRegProblem / t_overall_count * 100 << "%).";
				LOG(INFO) << "Registration: " << t_solve << " ms, (" << t_solve / t_overall_count * 100 << "%).";
				LOG(INFO) << "pub result: " << t_pub_result << " ms, (" << t_pub_result / t_overall_count * 100 << "%).";
				LOG(INFO) << "Total Computation (" << rpSolver_.lmStatics_.nPoints_ << "): " << t_overall_count << " ms.";
				LOG(INFO) << "------------------------------------------------------------";
				LOG(INFO) << "------------------------------------------------------------";
#endif
				if (bSaveTrajectory_)
				{
					struct stat st;
					if (stat(resultPath_.c_str(), &st) == -1) // there is no such dir, create one
					{
						LOG(INFO) << "There is no such directory: " << resultPath_;
						_mkdir(resultPath_.c_str());
						LOG(INFO) << "The directory has been created!!!";
					}
					LOG(INFO) << "pose size: " << lPose_.size();
					LOG(INFO) << "refPCMap_.size(): " << refPCMap_.size() << ", TS_history_.size(): " << TS_history_.size();
					saveTrajectory(resultPath_ + "result.txt");
				}
			}
			r.sleep();
		} // while
	}

	/**
    * @brief reload the current point cloud
    **/
	bool esvo_Tracking::refDataTransferring()
	{
		// load reference info
		ref_.t_ = refPCMap_buf_.back().first;
		if (ESVO_System_Status_ == "INITIALIZATION" && ets_ == IDLE) // will be true if receive the first PCMap from mapping
			ref_.tr_.setIdentity();
		if (ESVO_System_Status_ == "WORKING" || (ESVO_System_Status_ == "INITIALIZATION" && ets_ == WORKING))
		{
			if (!getPoseAt(ref_.t_, ref_.tr_, dvs_frame_id_))
			{
				LOG(INFO) << "ESVO_System_Status_: " << ESVO_System_Status_ << ", ref_.t_: " << ref_.t_.toNSec();
				LOG(INFO) << "Logic error ! There must be a pose for the given timestamp, because mapping has been finished.";
				exit(-1);
				return false;
			}
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
	bool esvo_Tracking::curDataTransferring()
	{
		// load current observation
		auto ev_last_it = EventBuffer_lower_bound(events_left_, cur_.t_);
		cur_.t_ = TS_buf_.back().first;
		cur_.pTsObs_ = &TS_buf_.back().second;
		cur_.tr_ = Transformation(T_world_cur_);
		auto ev_cur_it = EventBuffer_lower_bound(events_left_, cur_.t_);
		cur_.numEventsSinceLastObs_ = std::distance(ev_last_it, ev_cur_it) + 1; // Count the number of events occuring since the last observation.
		// LOG(INFO) << "event number in 10ms: " << cur_.numEventsSinceLastObs_; // 2000-1400
		return true;
	}

	void esvo_Tracking::reset()
	{
		m_buf_.lock();
		// clear all maintained data
		ets_ = IDLE;
		TS_id_ = 0;
		TS_history_.clear();
		TS_buf_.clear();
		refPCMap_.clear();
		refPCMap_buf_.clear();
		events_left_.clear();

		path_.poses.clear();
		path_gt_.poses.clear();
		m_buf_.unlock();
	}

	/********************** Callback functions *****************************/
	void esvo_Tracking::refMapCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
	{
		m_buf_.lock();
		pcl::PCLPointCloud2 pcl_pc;
		pcl_conversions::toPCL(*msg, pcl_pc);
		PointCloud::Ptr PC_ptr(new PointCloud());
		pcl::fromPCLPointCloud2(pcl_pc, *PC_ptr);
		refPCMap_buf_.push_back(std::make_pair(msg->header.stamp, PC_ptr));
		while (refPCMap_buf_.size() > REF_HISTORY_LENGTH_) // 10
			refPCMap_buf_.pop_front();
		m_buf_.unlock();
	}

	void esvo_Tracking::eventsCallback(const dvs_msgs::EventArray::ConstPtr &msg)
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
		m_buf_.unlock();
	}

	void esvo_Tracking::clearEventQueue()
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

	void esvo_Tracking::timeSurfaceCallback(const sensor_msgs::ImageConstPtr &time_surface_left,
											const sensor_msgs::ImageConstPtr &time_surface_right)
	{
		// std::lock_guard<std::mutex> lock(data_mutex_);
		cv_bridge::CvImagePtr cv_ptr_left, cv_ptr_right;
		try
		{
			cv_ptr_left = cv_bridge::toCvCopy(time_surface_left, sensor_msgs::image_encodings::MONO8);
			cv_ptr_right = cv_bridge::toCvCopy(time_surface_right, sensor_msgs::image_encodings::MONO8);
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

	void esvo_Tracking::publishTimeSurface()
	{
		if (TS_buf_.empty())
			return;
		time_surface_left_pub_.publish(TS_buf_.back().second.cvImagePtr_left_->toImageMsg());
		time_surface_right_pub_.publish(TS_buf_.back().second.cvImagePtr_right_->toImageMsg());
	}



	void esvo_Tracking::stampedPoseCallback(const geometry_msgs::PoseStampedConstPtr &msg)
	{
		m_buf_.lock();
		// add pose to tf
		tf::Transform tf(
			tf::Quaternion(
				msg->pose.orientation.x,
				msg->pose.orientation.y,
				msg->pose.orientation.z,
				msg->pose.orientation.w),
			tf::Vector3(
				msg->pose.position.x,
				msg->pose.position.y,
				msg->pose.position.z));
		tf::StampedTransform st(tf, msg->header.stamp, msg->header.frame_id, dvs_frame_id_.c_str());
		tf_->setTransform(st);
		// broadcast the tf such that the nav_path messages can find the valid fixed frame "map".
		static tf::TransformBroadcaster br;
		br.sendTransform(st);
		m_buf_.unlock();
	}

	bool esvo_Tracking::getPoseAt(const ros::Time &t, esvo_core::Transformation &Tr,
			  					  const std::string &source_frame)
	{
		std::string *err_msg = new std::string();
		if (!tf_->canTransform(world_frame_id_, source_frame, t, err_msg))
		{
			LOG(WARNING) << t.toNSec() << " : " << *err_msg;
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

	/************ publish results *******************/	
	void esvo_Tracking::gtPoseCallback(const geometry_msgs::PoseStampedConstPtr &msg)
	{
		// m_buf_.lock();
		// publish gt pose and path
		Eigen::Quaterniond q_gt_cur(msg->pose.orientation.w,
									msg->pose.orientation.x,
									msg->pose.orientation.y,
									msg->pose.orientation.z);
		Eigen::Vector3d t_gt_cur(msg->pose.position.x,
								 msg->pose.position.y,
								 msg->pose.position.z);
		if (path_gt_.poses.empty())
		{
			q_gt_s_ = Eigen::Quaterniond(msg->pose.orientation.w,
										 msg->pose.orientation.x,
										 msg->pose.orientation.y,
										 msg->pose.orientation.z);
			t_gt_s_ = Eigen::Vector3d(msg->pose.position.x,
									  msg->pose.position.y,
									  msg->pose.position.z);
		}
		Eigen::Quaterniond q_w_gt = q_gt_s_.inverse() * q_gt_cur;
		Eigen::Vector3d t_w_gt = q_gt_s_.inverse() * (t_gt_cur - t_gt_s_);

		geometry_msgs::PoseStampedPtr ps_ptr(new geometry_msgs::PoseStamped());
		ps_ptr->header.stamp = msg->header.stamp;
		ps_ptr->header.frame_id = world_frame_id_;
		ps_ptr->pose.orientation.x = q_w_gt.x();
		ps_ptr->pose.orientation.y = q_w_gt.y();
		ps_ptr->pose.orientation.z = q_w_gt.z();
		ps_ptr->pose.orientation.w = q_w_gt.w();
		ps_ptr->pose.position.x = t_w_gt.x();
		ps_ptr->pose.position.y = t_w_gt.y();
		ps_ptr->pose.position.z = t_w_gt.z();
		pose_gt_pub_.publish(ps_ptr);

		path_gt_.header.stamp = msg->header.stamp;
		path_gt_.header.frame_id = world_frame_id_;
		path_gt_.poses.push_back(*ps_ptr);
		path_gt_pub_.publish(path_gt_);

		tf::Transform tf(
			tf::Quaternion(
				q_w_gt.x(),
				q_w_gt.y(),
				q_w_gt.z(),
				q_w_gt.w()),
			tf::Vector3(
				t_w_gt.x(),
				t_w_gt.y(),
				t_w_gt.z()));
		tf::StampedTransform st(tf, msg->header.stamp, world_frame_id_, "gt");
		tf_->setTransform(st);
		// broadcast the tf such that the nav_path messages can find the valid fixed frame "map".
		static tf::TransformBroadcaster br;
		br.sendTransform(st);

		// m_buf_.unlock();
	}


	void esvo_Tracking::publishPose(const ros::Time &t, Transformation &tr)
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

	void esvo_Tracking::publishPath(const ros::Time &t, Transformation &tr)
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

	void
	esvo_Tracking::saveTrajectory(const std::string &resultDir)
	{
		LOG(INFO) << "Saving trajectory to " << resultDir << " ......";

		std::ofstream f;
		f.open(resultDir.c_str(), std::ofstream::out);
		if (!f.is_open())
		{
			LOG(INFO) << "File at " << resultDir << " is not opened, save trajectory failed.";
			exit(-1);
		}
		f << std::fixed;

		std::list<Eigen::Matrix<double, 4, 4>,
				  Eigen::aligned_allocator<Eigen::Matrix<double, 4, 4>>>::iterator result_it_begin = lPose_.begin();
		std::list<Eigen::Matrix<double, 4, 4>,
				  Eigen::aligned_allocator<Eigen::Matrix<double, 4, 4>>>::iterator result_it_end = lPose_.end();
		std::list<std::string>::iterator ts_it_begin = lTimestamp_.begin();

		for (; result_it_begin != result_it_end; result_it_begin++, ts_it_begin++)
		{
			Eigen::Matrix3d Rwc_result;
			Eigen::Vector3d twc_result;
			Rwc_result = (*result_it_begin).block<3, 3>(0, 0);
			twc_result = (*result_it_begin).block<3, 1>(0, 3);
			Eigen::Quaterniond q(Rwc_result);
			f << *ts_it_begin << " " << std::setprecision(9) << twc_result.transpose() << " "
			  << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
		}
		f.close();
		LOG(INFO) << "Saving trajectory to " << resultDir << ". Done !!!!!!.";
	}

} // namespace esvo_core