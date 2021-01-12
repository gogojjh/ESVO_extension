#include <esvo_core/estimator.h>
#include <esvo_core/DVS_MappingStereoConfig.h>
#include <esvo_core/tools/params_helper.h>

#include <minkindr_conversions/kindr_tf.h>

#include <geometry_msgs/TransformStamped.h>

#include <opencv2/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>

#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

#include <thread>
#include <iterator>
#include <memory>
#include <algorithm>
#include <utility>

//#define ESVO_CORE_MAPPING_DEBUG
//#define ESVO_CORE_MAPPING_LOG

namespace esvo_core
{
  esvo_Mapping::esvo_Mapping(
      const ros::NodeHandle &nh,
      const ros::NodeHandle &nh_private)
      : 
        // mapping parameters
        nh_(nh),
        pnh_(nh_private),
        TS_left_sub_(nh_, "time_surface_left", 10),
        TS_right_sub_(nh_, "time_surface_right", 10),
        TS_sync_(ExactSyncPolicy(10), TS_left_sub_, TS_right_sub_),
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
        dpSolver_(camSysPtr_, dpConfigPtr_, NUMERICAL, NUM_THREAD_MAPPING),
        dFusor_(camSysPtr_, dpConfigPtr_),
        dRegularizor_(dpConfigPtr_),
        ebm_(camSysPtr_, NUM_THREAD_MAPPING, tools::param(pnh_, "SmoothTimeSurface", false)),
        pc_(new PointCloud()),
        pc_near_(new PointCloud()),
        pc_global_(new PointCloud()),
        depthFramePtr_(new DepthFrame(camSysPtr_->cam_left_ptr_->height_, camSysPtr_->cam_left_ptr_->width_)),
        // tracking parameters
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
        rpSolver_(camSysPtr_, rpConfigPtr_, rpType_, NUM_THREAD_TRACKING)
  {
    // frame id
    dvs_frame_id_ = tools::param(pnh_, "dvs_frame_id", std::string("dvs"));
    world_frame_id_ = tools::param(pnh_, "world_frame_id", std::string("world"));
    pc_->header.frame_id = world_frame_id_;
    pc_near_->header.frame_id = world_frame_id_;
    pc_global_->header.frame_id = world_frame_id_;

    /**** mapping parameters ***/
    // range and visualization threshold
    invDepth_min_range_ = tools::param(pnh_, "invDepth_min_range", 0.16);
    invDepth_max_range_ = tools::param(pnh_, "invDepth_max_range", 2.0);
    patch_area_ = tools::param(pnh_, "patch_size_X", 25) * tools::param(pnh_, "patch_size_Y", 25);
    residual_vis_threshold_ = tools::param(pnh_, "residual_vis_threshold", 15);
    cost_vis_threshold_ = pow(residual_vis_threshold_, 2) * patch_area_;
    stdVar_vis_threshold_ = tools::param(pnh_, "stdVar_vis_threshold", 0.005);
    age_max_range_ = tools::param(pnh_, "age_max_range", 5);
    age_vis_threshold_ = tools::param(pnh_, "age_vis_threshold", 0);
    fusion_radius_ = tools::param(pnh_, "fusion_radius", 0);
    maxNumFusionFrames_ = tools::param(pnh_, "maxNumFusionFrames", 10);
    FusionStrategy_ = tools::param(pnh_, "FUSION_STRATEGY", std::string("CONST_FRAMES"));
    maxNumFusionPoints_ = tools::param(pnh_, "maxNumFusionPoints", 2000);
    INIT_SGM_DP_NUM_Threshold_ = tools::param(pnh_, "INIT_SGM_DP_NUM_THRESHOLD", 500);
    // options
    bDenoising_ = tools::param(pnh_, "Denoising", false);
    bRegularization_ = tools::param(pnh_, "Regularization", false);
    resetButton_ = tools::param(pnh_, "ResetButton", false);
    // visualization parameters
    bVisualizeGlobalPC_ = tools::param(pnh_, "bVisualizeGlobalPC", false);
    visualizeGPC_interval_ = tools::param(pnh_, "visualizeGPC_interval", 3);
    visualize_range_ = tools::param(pnh_, "visualize_range", 2.5);
    numAddedPC_threshold_ = tools::param(pnh_, "NumGPC_added_oper_refresh", 1000);
    // module parameters
    PROCESS_EVENT_NUM_ = tools::param(pnh_, "PROCESS_EVENT_NUM", 500);
    TS_HISTORY_LENGTH_ = tools::param(pnh_, "TS_HISTORY_LENGTH", 100);
    mapping_rate_hz_ = tools::param(pnh_, "mapping_rate_hz", 20);
    process_rate_hz_ = tools::param(pnh_, "process_rate_hz", 100);
    // Event Block Matching (BM) parameters
    BM_half_slice_thickness_ = tools::param(pnh_, "BM_half_slice_thickness", 0.001);
    BM_patch_size_X_ = tools::param(pnh_, "patch_size_X", 25);
    BM_patch_size_Y_ = tools::param(pnh_, "patch_size_Y", 25);
    BM_min_disparity_ = tools::param(pnh_, "BM_min_disparity", 3);
    BM_max_disparity_ = tools::param(pnh_, "BM_max_disparity", 40);
    BM_step_ = tools::param(pnh_, "BM_step", 1);
    BM_ZNCC_Threshold_ = tools::param(pnh_, "BM_ZNCC_Threshold", 0.1);
    BM_bUpDownConfiguration_ = tools::param(pnh_, "BM_bUpDownConfiguration", false);

    // SGM parameters (Used by Initialization)
    num_disparities_ = 16 * 3;
    block_size_ = 11;
    P1_ = 8 * 1 * block_size_ * block_size_;
    P2_ = 32 * 1 * block_size_ * block_size_;
    uniqueness_ratio_ = 11;
    sgbm_ = cv::StereoSGBM::create(0, num_disparities_, block_size_, P1_, P2_,
                                   -1, 0, uniqueness_ratio_);

    // calcualte the min,max disparity
    double f = (camSysPtr_->cam_left_ptr_->P_(0, 0) + camSysPtr_->cam_left_ptr_->P_(1, 1)) / 2;
    double b = camSysPtr_->baseline_;
    size_t minDisparity = max(size_t(std::floor(f * b * invDepth_min_range_)), (size_t)0);
    size_t maxDisparity = size_t(std::ceil(f * b * invDepth_max_range_));
    minDisparity = max(minDisparity, BM_min_disparity_);
    maxDisparity = min(maxDisparity, BM_max_disparity_);

#ifdef ESVO_CORE_MAPPING_DEBUG
    LOG(INFO) << "f: " << f << " "
              << " b: " << b;
    LOG(INFO) << "invDepth_min_range_: " << invDepth_min_range_;
    LOG(INFO) << "invDepth_max_range_: " << invDepth_max_range_;
    LOG(INFO) << "minDisparity: " << minDisparity;
    LOG(INFO) << "maxDisparity: " << maxDisparity;
#endif

    // initialize Event Batch Matcher
    ebm_.resetParameters(BM_patch_size_X_, BM_patch_size_Y_, minDisparity, maxDisparity,
                         BM_step_, BM_ZNCC_Threshold_, BM_bUpDownConfiguration_);

    // system status
    ESVO_System_Status_ = "INITIALIZATION";
    nh_.setParam("/ESVO_SYSTEM_STATUS", ESVO_System_Status_);

    // callback functions
    events_left_sub_ = nh_.subscribe<dvs_msgs::EventArray>("events_left", 0, boost::bind(&esvo_Mapping::eventsCallback, this, _1, boost::ref(events_left_)));
    events_right_sub_ = nh_.subscribe<dvs_msgs::EventArray>("events_right", 0, boost::bind(&esvo_Mapping::eventsCallback, this, _1, boost::ref(events_right_)));
    stampedPose_sub_ = nh_.subscribe("stamped_pose", 0, &esvo_Mapping::stampedPoseCallback, this);
    TS_sync_.registerCallback(boost::bind(&esvo_Mapping::timeSurfaceCallback, this, _1, _2));
    // TF
    tf_ = std::make_shared<tf::Transformer>(true, ros::Duration(100.0));

    // result publishers
    invDepthMap_pub_ = it_.advertise("Inverse_Depth_Map", 1);
    stdVarMap_pub_ = it_.advertise("Standard_Variance_Map", 1);
    ageMap_pub_ = it_.advertise("Age_Map", 1);
    costMap_pub_ = it_.advertise("cost_map", 1);
    pc_pub_ = nh_.advertise<PointCloud>("/esvo_mapping/pointcloud_local", 1);
    eventFrame_pub_ = it_.advertise("Event_Frame", 1);
    if (bVisualizeGlobalPC_)
    {
      gpc_pub_ = nh_.advertise<PointCloud>("/esvo_mapping/pointcloud_global", 1);
      pc_global_->reserve(500000);
      t_last_pub_pc_ = 0.0;
    }

    // multi-thread management
    mapping_thread_future_ = mapping_thread_promise_.get_future();
    reset_future_ = reset_promise_.get_future();

    // stereo mapping detached thread
    std::thread MappingThread(&esvo_Mapping::Process, this,
                              std::move(mapping_thread_promise_), std::move(reset_future_));
    MappingThread.detach();

    // Dynamic reconfigure
    dynamic_reconfigure_callback_ = boost::bind(&esvo_Mapping::onlineParameterChangeCallback, this, _1, _2);
    server_.reset(new dynamic_reconfigure::Server<DVS_MappingStereoConfig>(nh_private));
    server_->setCallback(dynamic_reconfigure_callback_);

    invDepth_INIT_ = 1.0;

    reprojMap_pub_left_ = it_.advertise("Reproj_Map_Left", 1);
    rpSolver_.setRegPublisher(&reprojMap_pub_left_);

    pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/esvo_tracking/pose_pub", 1);
    path_pub_ = nh_.advertise<nav_msgs::Path>("/esvo_tracking/trajectory", 1);

    bVisualizeTrajectory_ = true;
    
    T_world_cur_ = Eigen::Matrix<double, 4, 4>::Identity();
  }

  esvo_Mapping::~esvo_Mapping()
  {
    pc_pub_.shutdown();
    invDepthMap_pub_.shutdown();
    stdVarMap_pub_.shutdown();
    ageMap_pub_.shutdown();
    costMap_pub_.shutdown();
  }

  void esvo_Mapping::Process(
      std::promise<void> prom_mapping,
      std::future<void> future_reset)
  {
    ros::Rate r(process_rate_hz_);
    while (ros::ok())
    {
      // reset mapping rate
      if (changed_frame_rate_)
      {
        r = ros::Rate(process_rate_hz_);
        changed_frame_rate_ = false;
      }

      // check system status
      nh_.getParam("/ESVO_SYSTEM_STATUS", ESVO_System_Status_);
      // LOG(INFO) << "SYSTEM STATUS (Process): " << ESVO_System_Status_;

      if (ESVO_System_Status_ == "TERMINATE")
      {
        LOG(INFO) << "The Mapping node is terminated manually...";
        break;
      }

      if (TS_history_.size() >= 10) /* To assure the esvo_time_surface node has been working. */
      {
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

#ifdef MONOCULAR_DEBUG
        if (ESVO_System_Status_ == "INITIALIZATION" || ESVO_System_Status_ == "RESET")
        {
          if (MonoInitializationAtTime(TS_obs_.first))
          {
            LOG(INFO) << "Initialization is successfully done!"; //(" << INITIALIZATION_COUNTER_ << ").";
            nh_.setParam("/ESVO_SYSTEM_STATUS", "WORKING");
          }
          else
            LOG(INFO) << "Initialization fails once.";
        }
        else if (ESVO_System_Status_ == "WORKING") 
        {
          MappingAtTime(TS_obs_.first);
        }
        std::thread tPublishEventFrame(&esvo_Mapping::publishEventFrame, this, TS_obs_.first);
        tPublishEventFrame.detach();
#else
        if (ESVO_System_Status_ == "INITIALIZATION" || ESVO_System_Status_ == "RESET") // do initialization
        {
          if (InitializationAtTime(TS_obs_.first))
          {
            LOG(INFO) << "Initialization is successfully done!"; //(" << INITIALIZATION_COUNTER_ << ").";
            nh_.setParam("/ESVO_SYSTEM_STATUS", "WORKING");
          }
          else
            LOG(INFO) << "Initialization fails once.";
        }
        
        if (ESVO_System_Status_ == "WORKING") // do tracking & mapping
        {
          // tracking
          setTrackingData();
          LOG(INFO) << "set tracking data: ref, cur";
          if (rpSolver_.resetRegProblem(&ref_, &cur_))
          {
            TicToc t_solve;
            if (rpType_ == REG_NUMERICAL)
              rpSolver_.solve_numerical();
            if (rpType_ == REG_ANALYTICAL)
              rpSolver_.solve_analytical();
            LOG(INFO) << "tracking costs " <<  t_solve.toc() << "ms";
            T_world_cur_ = cur_.tr_.getTransformationMatrix();
            std::cout << "tracking pose: " << std::endl;
            std::cout << T_world_cur_ << std::endl;
          }
          Transformation tr_world_cur(T_world_cur_);
          TS_obs_.second.setTransformation(tr_world_cur);
          publishPose(cur_.t_, cur_.tr_);
          if (bVisualizeTrajectory_)
          {
            publishPath(cur_.t_, cur_.tr_);
            // save results to listPose and listPoseGt
            lTimestamp_.push_back(std::to_string(cur_.t_.toSec()));
            lPose_.push_back(cur_.tr_.getTransformationMatrix());
          }

          // mapping
          // add pose to tf
          Eigen::Matrix3d R_world_cur = T_world_cur_.topLeftCorner<3, 3>();
          Eigen::Quaterniond q_world_cur(R_world_cur);
          Eigen::Vector3d t_world_cur = T_world_cur_.topRightCorner<3, 1>();
          tf::Transform tf(
              tf::Quaternion(
                  q_world_cur.x(),
                  q_world_cur.y(),
                  q_world_cur.z(),
                  q_world_cur.w()),
              tf::Vector3(
                  t_world_cur.x(),
                  t_world_cur.y(),
                  t_world_cur.z()));
          tf::StampedTransform st(tf, cur_.t_, "world", dvs_frame_id_.c_str());
          tf_->setTransform(st);

          st_map_.clear();
          ros::Time t_end = TS_obs_.first;
          ros::Time t_begin(std::max(0.0, t_end.toSec() - 10 * BM_half_slice_thickness_));
          ros::Time t_tmp = t_begin;
          while (t_tmp.toSec() <= t_end.toSec())
          {
            Transformation tr;
            if (getPoseAt(t_tmp, tr, dvs_frame_id_))
            {
              st_map_.emplace(t_tmp, tr);
              printf("interpolation: %f\n", t_tmp.toSec());
              std::cout << tr.getTransformationMatrix() << std::endl;
            }
            t_tmp = ros::Time(t_tmp.toSec() + 0.05 * BM_half_slice_thickness_);
          }

          // if (mapping_rate_hz_)
          MappingAtTime(TS_obs_.first);
        }
#endif

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

  // extract depthMap from depthFramePtr_->dMap_
  bool esvo_Mapping::setTrackingData()
  {
    // set map
    DepthMap::Ptr depthMapPtr = depthFramePtr_->dMap_;
    Transformation tr = depthFramePtr_->T_world_frame_;

    Eigen::Matrix<double, 4, 4> T_world_result = tr.getTransformationMatrix();
    PointCloud::Ptr pc_local(new PointCloud());
    pc_local->clear();
    pc_local->reserve(50000);
    for (auto it = depthMapPtr->begin(); it != depthMapPtr->end(); it++)
    {
      Eigen::Vector3d p_world = T_world_result.block<3, 3>(0, 0) * it->p_cam() + T_world_result.block<3, 1>(0, 3);
      pc_local->push_back(pcl::PointXYZ(p_world(0), p_world(1), p_world(2)));
    }

    size_t numPoint = pc_local->size();
    ref_.vPointXYZPtr_.clear();
    ref_.vPointXYZPtr_.reserve(numPoint);
    auto PointXYZ_begin_it = pc_local->begin();
    auto PointXYZ_end_it = pc_local->end();
    while (PointXYZ_begin_it != PointXYZ_end_it)
    {
      ref_.vPointXYZPtr_.push_back(PointXYZ_begin_it.base()); // Copy the pointer of the pointXYZ
      PointXYZ_begin_it++;
    }
    LOG(INFO) << "map point size: " << pc_local->size();

    // set TS
    cur_.t_ = TS_obs_.first;
    cur_.pTsObs_ = &TS_obs_.second;
    cur_.tr_ = Transformation(T_world_cur_);
    cur_.numEventsSinceLastObs_ = vALLEventsPtr_left_.size();
    LOG(INFO) << "current event size: " << cur_.numEventsSinceLastObs_;

    return true;
  }

  // follow LSD-SLAM to initialize the map
  bool esvo_Mapping::MonoInitializationAtTime(const ros::Time &t)
  {
    cv::Mat edgeMap;
    std::vector<std::pair<size_t, size_t>> vEdgeletCoordinates;
    createEdgeMask(vALLEventsPtr_left_, camSysPtr_->cam_left_ptr_,
                   edgeMap, vEdgeletCoordinates, true, 0);

    std::vector<DepthPoint> vdp;
    vdp.reserve(vEdgeletCoordinates.size());
    double var_SGM = VAR_RANDOM_INIT_INITIAL_;
    for (size_t i = 0; i < vEdgeletCoordinates.size(); i++)
    {
      size_t x = vEdgeletCoordinates[i].first;
      size_t y = vEdgeletCoordinates[i].second;
      DepthPoint dp(x, y);
      Eigen::Vector2d p_img(x * 1.0, y * 1.0);
      dp.update_x(p_img);
      double invDepth = 1.0 / (0.5 + 1.0 * ((rand() % 100001) / 100000.0));

      Eigen::Vector3d p_cam;
      camSysPtr_->cam_left_ptr_->cam2World(p_img, invDepth, p_cam);
      dp.update_p_cam(p_cam);
      dp.update(invDepth, var_SGM);
      dp.residual() = 0.0;
      dp.age() = age_vis_threshold_;
      Eigen::Matrix<double, 4, 4> T_world_cam = TS_obs_.second.tr_.getTransformationMatrix();
      dp.updatePose(T_world_cam);
      vdp.push_back(dp);
    }
    LOG(INFO) << "********** Initialization returns " << vdp.size() << " points.";
    if (vdp.size() < INIT_DP_NUM_Threshold_)
      return false;

    // push the "masked" SGM results to the depthFrame
    dqvDepthPoints_.push_back(vdp);
    dFusor_.naive_propagation(vdp, depthFramePtr_);

    // publish the invDepth map
    std::thread tPublishMappingResult(&esvo_Mapping::publishMappingResults, this,
                                      depthFramePtr_->dMap_, depthFramePtr_->T_world_frame_, t);
    tPublishMappingResult.detach();
    return true;
  }

  bool esvo_Mapping::InitializationAtTime(const ros::Time &t)
  {
    // create a new depth frame
    DepthFrame::Ptr depthFramePtr_new = std::make_shared<DepthFrame>(
        camSysPtr_->cam_left_ptr_->height_, camSysPtr_->cam_left_ptr_->width_);
    depthFramePtr_new->setId(TS_obs_.second.id_);
    depthFramePtr_new->setTransformation(TS_obs_.second.tr_); // identity
    depthFramePtr_ = depthFramePtr_new;

    // call SGM on the current Time Surface observation pair.
    cv::Mat dispMap, dispMap8;
    sgbm_->compute(TS_obs_.second.cvImagePtr_left_->image, TS_obs_.second.cvImagePtr_right_->image, dispMap);
    dispMap.convertTo(dispMap8, CV_8U, 255 / (num_disparities_ * 16.));

    // get the event map (binary mask)
    cv::Mat edgeMap;
    std::vector<std::pair<size_t, size_t>> vEdgeletCoordinates;
    createEdgeMask(vEventsPtr_left_SGM_, camSysPtr_->cam_left_ptr_,
                   edgeMap, vEdgeletCoordinates, true, 0);

    // Apply logical "AND" operation and transfer "disparity" to "invDepth".
    std::vector<DepthPoint> vdp_sgm;
    vdp_sgm.reserve(vEdgeletCoordinates.size());
    double var_SGM = pow(stdVar_vis_threshold_ * 0.99, 2);
    for (size_t i = 0; i < vEdgeletCoordinates.size(); i++)
    {
      size_t x = vEdgeletCoordinates[i].first;
      size_t y = vEdgeletCoordinates[i].second;

      double disp = dispMap.at<short>(y, x) / 16.0;
      if (disp < 0)
        continue;
      DepthPoint dp(x, y);
      Eigen::Vector2d p_img(x * 1.0, y * 1.0);
      dp.update_x(p_img);
      double invDepth = disp / (camSysPtr_->cam_left_ptr_->P_(0, 0) * camSysPtr_->baseline_);
      if (invDepth < invDepth_min_range_ || invDepth > invDepth_max_range_)
        continue;
      Eigen::Vector3d p_cam;
      camSysPtr_->cam_left_ptr_->cam2World(p_img, invDepth, p_cam);
      dp.update_p_cam(p_cam);
      dp.update(invDepth, var_SGM);
      dp.residual() = 0.0;
      dp.age() = age_vis_threshold_;
      Eigen::Matrix<double, 4, 4> T_world_cam = TS_obs_.second.tr_.getTransformationMatrix();
      dp.updatePose(T_world_cam);
      vdp_sgm.push_back(dp);
    }
    LOG(INFO) << "********** Initialization (SGM) returns " << vdp_sgm.size() << " points.";
    if (vdp_sgm.size() < INIT_SGM_DP_NUM_Threshold_)
      return false;

    // push the "masked" SGM results to the depthFrame
    dqvDepthPoints_.push_back(vdp_sgm);
    dFusor_.naive_propagation(vdp_sgm, depthFramePtr_);

    // publish the invDepth map
    std::thread tPublishMappingResult(&esvo_Mapping::publishMappingResults, this,
                                      depthFramePtr_->dMap_, depthFramePtr_->T_world_frame_, t);
    tPublishMappingResult.detach();
    return true;
  }

  void esvo_Mapping::MappingAtTime(const ros::Time &t)
  {
    TicToc tt_mapping;
    double t_overall_count = 0;
    /************************************************/
    /************ set the new DepthFrame ************/
    /************************************************/
    DepthFrame::Ptr depthFramePtr_new = std::make_shared<DepthFrame>(
        camSysPtr_->cam_left_ptr_->height_, camSysPtr_->cam_left_ptr_->width_);
    depthFramePtr_new->setId(TS_obs_.second.id_);
    depthFramePtr_new->setTransformation(TS_obs_.second.tr_);
    depthFramePtr_ = depthFramePtr_new;

    std::vector<EventMatchPair> vEMP; // the container that stores the result of BM.
    /****************************************************/
    /*************** Block Matching (BM) ****************/
    /****************************************************/
    double t_BM = 0.0;
    double t_BM_denoising = 0.0;

    // Denoising operations
    if (bDenoising_) // Set it to "True" to deal with flicker effect caused by VICON.
    {
      tt_mapping.tic();
      // Draw one mask image for denoising.
      cv::Mat denoising_mask;
      createDenoisingMask(vALLEventsPtr_left_, denoising_mask,
                          camSysPtr_->cam_left_ptr_->height_, camSysPtr_->cam_left_ptr_->width_);

      // Extract denoised events (appear on edges likely).
      vDenoisedEventsPtr_left_.clear();
      extractDenoisedEvents(vCloseEventsPtr_left_, vDenoisedEventsPtr_left_, denoising_mask, PROCESS_EVENT_NUM_);
      totalNumCount_ = vDenoisedEventsPtr_left_.size();

      t_BM_denoising = tt_mapping.toc();
    }
    else
    {
      vDenoisedEventsPtr_left_.clear();
      vDenoisedEventsPtr_left_.reserve(PROCESS_EVENT_NUM_);
      vDenoisedEventsPtr_left_.insert(
          vDenoisedEventsPtr_left_.end(), vCloseEventsPtr_left_.begin(),
          vCloseEventsPtr_left_.begin() + min(vCloseEventsPtr_left_.size(), PROCESS_EVENT_NUM_));
    }
    // LOG(INFO) << "Denosing succeeds";

    // block matching
    tt_mapping.tic();
    ebm_.createMatchProblem(&TS_obs_, &st_map_, &vDenoisedEventsPtr_left_);
    ebm_.match_all_HyperThread(vEMP);
    // ebm_.match_all_SingleThread(vEMP);
#ifdef ESVO_CORE_MAPPING_DEBUG
    LOG(INFO) << "++++ Block Matching (BM) generates " << vEMP.size() << " candidates.";
#endif
    t_BM = tt_mapping.toc();
    t_overall_count += t_BM_denoising;
    t_overall_count += t_BM;
    // LOG(INFO) << "Block matching succeeds";

    /**************************************************************/
    /*************  Nonlinear Optimization & Fusion ***************/
    /**************************************************************/
    double t_optimization = 0;
    double t_solve, t_fusion, t_regularization;
    t_solve = t_fusion = t_regularization = 0;
    size_t numFusionCount = 0; // To count the total number of fusion (in terms of fusion between two estimates, i.e. a priori and a propagated one).

    tt_mapping.tic();

    // Nonlinear opitmization
    std::vector<DepthPoint> vdp;
    vdp.reserve(vEMP.size());
    dpSolver_.solve(&vEMP, &TS_obs_, vdp); // hyper-thread version
#ifdef ESVO_CORE_MAPPING_DEBUG
    LOG(INFO) << "Nonlinear optimization returns: " << vdp.size() << " estimates.";
#endif
    // culling points by checking its std, cost, inverse depth
    dpSolver_.pointCulling(vdp, stdVar_vis_threshold_, cost_vis_threshold_,
                           invDepth_min_range_, invDepth_max_range_); 
#ifdef ESVO_CORE_MAPPING_DEBUG
    LOG(INFO) << "After culling, vdp.size: " << vdp.size();
#endif
    t_solve = tt_mapping.toc();
    LOG(INFO) << "Nonliner optimization succeeds";

    // Fusion (strategy 1: const number of point)
    if (FusionStrategy_ == "CONST_POINTS")
    {
      size_t numFusionPoints = 0;
      tt_mapping.tic();
      dqvDepthPoints_.push_back(vdp);
      for (size_t n = 0; n < dqvDepthPoints_.size(); n++)
        numFusionPoints += dqvDepthPoints_[n].size();
      while (numFusionPoints > 1.5 * maxNumFusionPoints_)
      {
        dqvDepthPoints_.pop_front();
        numFusionPoints = 0;
        for (size_t n = 0; n < dqvDepthPoints_.size(); n++)
          numFusionPoints += dqvDepthPoints_[n].size();
      }
    }
    else if (FusionStrategy_ == "CONST_FRAMES") // (strategy 2: const number of frames)
    {
      tt_mapping.tic();
      dqvDepthPoints_.push_back(vdp);
      while (dqvDepthPoints_.size() > maxNumFusionFrames_)
        dqvDepthPoints_.pop_front();
    }
    else
    {
      LOG(INFO) << "Invalid FusionStrategy is assigned.";
      exit(-1);
    }
    LOG(INFO) << "Fusion succeeds";

    // apply fusion and count the total number of fusion.
    numFusionCount = 0;
    for (auto it = dqvDepthPoints_.rbegin(); it != dqvDepthPoints_.rend(); it++)
    {
      numFusionCount += dFusor_.update(*it, depthFramePtr_, fusion_radius_); // fusing new points to update the depthmap
      //    LOG(INFO) << "numFusionCount: " << numFusionCount;
    }

    TotalNumFusion_ += numFusionCount;
    depthFramePtr_->dMap_->clean(pow(stdVar_vis_threshold_, 2), age_vis_threshold_, invDepth_max_range_, invDepth_min_range_);
    t_fusion = tt_mapping.toc();

    // regularization
    if (bRegularization_)
    {
      tt_mapping.tic();
      dRegularizor_.apply(depthFramePtr_->dMap_);
      t_regularization = tt_mapping.toc();
    }
    // count time
    t_optimization = t_solve + t_fusion + t_regularization;
    t_overall_count += t_optimization;

    // publish results
    std::thread tPublishMappingResult(&esvo_Mapping::publishMappingResults, this,
                                      depthFramePtr_->dMap_, depthFramePtr_->T_world_frame_, t);
    tPublishMappingResult.detach();

#ifdef ESVO_CORE_MAPPING_LOG
    LOG(INFO) << "\n";
    LOG(INFO) << "------------------------------------------------------------";
    LOG(INFO) << "--------------Computation Cost (Mapping)---------------------";
    LOG(INFO) << "------------------------------------------------------------";
    LOG(INFO) << "Denoising: " << t_BM_denoising << " ms, (" << t_BM_denoising / t_overall_count * 100 << "%).";
    LOG(INFO) << "Block Matching (BM): " << t_BM << " ms, (" << t_BM / t_overall_count * 100 << "%).";
    LOG(INFO) << "BM success ratio: " << vEMP.size() << "/" << totalNumCount_ << "(Successes/Total).";
    LOG(INFO) << "------------------------------------------------------------";
    LOG(INFO) << "------------------------------------------------------------";
    LOG(INFO) << "Update: " << t_optimization << " ms, (" << t_optimization / t_overall_count * 100
              << "%).";
    LOG(INFO) << "-- nonlinear optimization: " << t_solve << " ms, (" << t_solve / t_overall_count * 100
              << "%).";
    LOG(INFO) << "-- fusion (" << numFusionCount << ", " << TotalNumFusion_ << "): " << t_fusion << " ms, (" << t_fusion / t_overall_count * 100
              << "%).";
    LOG(INFO) << "-- regularization: " << t_regularization << " ms, (" << t_regularization / t_overall_count * 100
              << "%).";
    LOG(INFO) << "------------------------------------------------------------";
    LOG(INFO) << "------------------------------------------------------------";
    LOG(INFO) << "Total Computation (" << depthFramePtr_->dMap_->size() << "): " << t_overall_count << " ms.";
    LOG(INFO) << "------------------------------------------------------------";
    LOG(INFO) << "------------------------------END---------------------------";
    LOG(INFO) << "------------------------------------------------------------";
    LOG(INFO) << "\n";
#endif
  }

  bool esvo_Mapping::dataTransferring()
  {
    TS_obs_ = std::make_pair(ros::Time(), TimeSurfaceObservation()); // clean the TS obs.
    if (TS_history_.size() <= 10)                                    /* To assure the esvo_time_surface node has been working. */
      return false;
    totalNumCount_ = 0;

#ifdef MONOCULAR_DEBUG
    auto it_end = TS_history_.rbegin();
    TS_obs_ = *it_end;

    if (ESVO_System_Status_ == "INITIALIZATION")
    {
      // copy all involved events' pointers
      vALLEventsPtr_left_.clear();   // Used to generate denoising mask (only used to deal with flicker induced by VICON.)

      // load allEvent
      ros::Time t_end = TS_obs_.first;
      ros::Time t_begin(std::max(0.0, t_end.toSec() - 0.002)); // 10ms
      auto ev_end_it = tools::EventBuffer_lower_bound(events_left_, t_end);
      auto ev_begin_it = tools::EventBuffer_lower_bound(events_left_, t_begin);
      const size_t MAX_NUM_Event_INVOLVED = 30000;
      vALLEventsPtr_left_.reserve(MAX_NUM_Event_INVOLVED);
      while (ev_end_it != ev_begin_it && vALLEventsPtr_left_.size() < MAX_NUM_Event_INVOLVED)
      {
        vALLEventsPtr_left_.push_back(ev_end_it._M_cur);
        ev_end_it--;
      }
    } 
    else if (ESVO_System_Status_ == "WORKING")
    {
      // copy all involved events' pointers
      vALLEventsPtr_left_.clear();   // Used to generate denoising mask (only used to deal with flicker induced by VICON.)
      vCloseEventsPtr_left_.clear(); // Will be denoised using the mask above.

      // load allEvent
      ros::Time t_end = TS_obs_.first;
      ros::Time t_begin(std::max(0.0, t_end.toSec() - 0.01)); // 10ms
      auto ev_end_it = tools::EventBuffer_lower_bound(events_left_, t_end);
      auto ev_begin_it = tools::EventBuffer_lower_bound(events_left_, t_begin);
      const size_t MAX_NUM_Event_INVOLVED = 10000;
      vALLEventsPtr_left_.reserve(MAX_NUM_Event_INVOLVED);
      vCloseEventsPtr_left_.reserve(MAX_NUM_Event_INVOLVED);
      while (ev_end_it != ev_begin_it && vALLEventsPtr_left_.size() < MAX_NUM_Event_INVOLVED)
      {
        vALLEventsPtr_left_.push_back(ev_end_it._M_cur);
        vCloseEventsPtr_left_.push_back(ev_end_it._M_cur);
        ev_end_it--;
      }
    }
    totalNumCount_ = vALLEventsPtr_left_.size();

#ifdef ESTIMATOR_DEBUG
    LOG(INFO) << "Data Transferring (vALLEventsPtr_left_): " << vALLEventsPtr_left_.size();
    LOG(INFO) << "Data Transforming (vCloseEventsPtr_left_): " << vCloseEventsPtr_left_.size();
#endif
    return true;
#else
    // load current Time-Surface Observation
    auto it_end = TS_history_.rbegin();
    it_end++; // in case that the tf is behind the most current TS.
    auto it_begin = TS_history_.begin();
    while (TS_obs_.second.isEmpty())
    {
      Transformation tr;
      if (ESVO_System_Status_ == "INITIALIZATION")
      {
        tr.setIdentity();
        it_end->second.setTransformation(tr);
        TS_obs_ = *it_end;
      }
      else if (ESVO_System_Status_ == "WORKING")
      {
        // if (getPoseAt(it_end->first, tr, dvs_frame_id_))
        // {
        //   it_end->second.setTransformation(tr);
        //   TS_obs_ = *it_end;
        // }
        // else
        // {
        //   // check if the tracking node is still working normally
        //   nh_.getParam("/ESVO_SYSTEM_STATUS", ESVO_System_Status_);
        //   if (ESVO_System_Status_ != "WORKING")
        //     return false;
        // }
        TS_obs_ = *it_end;
      }
      if (it_end->first == it_begin->first)
        break;
      it_end++;
    }
    if (TS_obs_.second.isEmpty())
      return false;

    /****** Load involved events *****/
    // SGM
    if (ESVO_System_Status_ == "INITIALIZATION")
    {
      vEventsPtr_left_SGM_.clear();
      ros::Time t_end = TS_obs_.first;
      ros::Time t_begin(std::max(0.0, t_end.toSec() - 2 * BM_half_slice_thickness_)); // 2ms
      auto ev_end_it = tools::EventBuffer_lower_bound(events_left_, t_end);
      auto ev_begin_it = tools::EventBuffer_lower_bound(events_left_, t_begin);
      const size_t MAX_NUM_Event_INVOLVED = 30000;
      vEventsPtr_left_SGM_.reserve(MAX_NUM_Event_INVOLVED);
      while (ev_end_it != ev_begin_it && vEventsPtr_left_SGM_.size() <= PROCESS_EVENT_NUM_)
      {
        vEventsPtr_left_SGM_.push_back(ev_end_it._M_cur);
        ev_end_it--;
      }
    }

    // BM
    if (ESVO_System_Status_ == "WORKING")
    {
      // copy all involved events' pointers
      vALLEventsPtr_left_.clear();   // Used to generate denoising mask (only used to deal with flicker induced by VICON.)
      vCloseEventsPtr_left_.clear(); // Will be denoised using the mask above.

      // load allEvent
      ros::Time t_end = TS_obs_.first;
      ros::Time t_begin(std::max(0.0, t_end.toSec() - 10 * BM_half_slice_thickness_)); // 10ms
      auto ev_end_it = tools::EventBuffer_lower_bound(events_left_, t_end);
      auto ev_begin_it = tools::EventBuffer_lower_bound(events_left_, t_begin);
      const size_t MAX_NUM_Event_INVOLVED = 10000;
      vALLEventsPtr_left_.reserve(MAX_NUM_Event_INVOLVED);
      vCloseEventsPtr_left_.reserve(MAX_NUM_Event_INVOLVED);
      while (ev_end_it != ev_begin_it && vALLEventsPtr_left_.size() < MAX_NUM_Event_INVOLVED)
      {
        vALLEventsPtr_left_.push_back(ev_end_it._M_cur);
        vCloseEventsPtr_left_.push_back(ev_end_it._M_cur);
        ev_end_it--;
      }
      totalNumCount_ = vCloseEventsPtr_left_.size();
      
#ifdef ESVO_CORE_MAPPING_DEBUG
      LOG(INFO) << "Data Transferring (vALLEventsPtr_left_): " << vALLEventsPtr_left_.size();
      LOG(INFO) << "Data Transforming (vCloseEventsPtr_left_): " << vCloseEventsPtr_left_.size();
#endif

      // Ideally, each event occurs at an unique perspective (virtual view) -- pose.
      // In practice, this is intractable in real-time application.
      // We made a trade off by assuming that events occurred within (0.05 * BM_half_slice_thickness_) ms share an identical pose (virtual view).
      // Here we load transformations for all virtual views.
      // st_map_.clear();
      // ros::Time t_tmp = t_begin;
      // while (t_tmp.toSec() <= t_end.toSec())
      // {
      //   Transformation tr;
      //   if (getPoseAt(t_tmp, tr, dvs_frame_id_))
      //     st_map_.emplace(t_tmp, tr);
      //   else
      //   {
      //     nh_.getParam("/ESVO_SYSTEM_STATUS", ESVO_System_Status_);
      //     if (ESVO_System_Status_ != "WORKING")
      //       return false;
      //   }
      //   t_tmp = ros::Time(t_tmp.toSec() + 0.05 * BM_half_slice_thickness_);
      // }
#ifdef ESVO_CORE_MAPPING_DEBUG
      LOG(INFO) << "Data Transferring (stampTransformation map): " << st_map_.size();
#endif

#endif
    }
    return true;
  }

  void esvo_Mapping::stampedPoseCallback(const geometry_msgs::PoseStampedConstPtr &ps_msg)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    // To check inconsistent timestamps and reset.
    static constexpr double max_time_diff_before_reset_s = 0.5;
    const ros::Time stamp_first_event = ps_msg->header.stamp;
    std::string *err_tf = new std::string();
    //  int iGetLastest_common_time =
    //    tf_->getLatestCommonTime(dvs_frame_id_.c_str(), ps_msg->header.frame_id, tf_lastest_common_time_, err_tf);
    delete err_tf;

    if (tf_lastest_common_time_.toSec() != 0)
    {
      const double dt = stamp_first_event.toSec() - tf_lastest_common_time_.toSec();
      if (dt < 0 || std::fabs(dt) >= max_time_diff_before_reset_s)
      {
        ROS_INFO("Inconsistent event timestamps detected <stampedPoseCallback> (new: %f, old %f), resetting.",
                 stamp_first_event.toSec(), tf_lastest_common_time_.toSec());
        reset();
      }
    }

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
  }

  // return the pose of the left event cam at time t.
  bool esvo_Mapping::getPoseAt(
      const ros::Time &t,
      esvo_core::Transformation &Tr, // T_world_virtual
      const std::string &source_frame)
  {
    std::string *err_msg = new std::string();
    if (!tf_->canTransform(world_frame_id_, source_frame, t, err_msg))
    {
#ifdef ESVO_CORE_MAPPING_LOG
      LOG(WARNING) << t.toNSec() << " : " << *err_msg;
#endif
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

  void esvo_Mapping::eventsCallback(
      const dvs_msgs::EventArray::ConstPtr &msg,
      EventQueue &EQ)
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
        ROS_INFO("Inconsistent event timestamps detected <eventCallback> (new: %f, old %f), resetting.",
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

  void
  esvo_Mapping::clearEventQueue(EventQueue &EQ)
  {
    static constexpr size_t MAX_EVENT_QUEUE_LENGTH = 3000000;
    if (EQ.size() > MAX_EVENT_QUEUE_LENGTH)
    {
      size_t NUM_EVENTS_TO_REMOVE = EQ.size() - MAX_EVENT_QUEUE_LENGTH;
      EQ.erase(EQ.begin(), EQ.begin() + NUM_EVENTS_TO_REMOVE);
    }
  }

  void esvo_Mapping::timeSurfaceCallback(
      const sensor_msgs::ImageConstPtr &time_surface_left,
      const sensor_msgs::ImageConstPtr &time_surface_right)
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
        ROS_INFO("Inconsistent frame timestamp detected <timeSurfaceCallback> (new: %f, old %f), resetting.",
                 time_surface_left->header.stamp.toSec(), stamp_last_image.toSec());
        reset();
      }
    }

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

    // push back the new time surface map
    ros::Time t_new_TS = time_surface_left->header.stamp;
    // Made the gradient computation optional which is up to the jacobian choice.
    if (dpSolver_.getProblemType() == NUMERICAL)
      TS_history_.emplace(t_new_TS, TimeSurfaceObservation(cv_ptr_left, cv_ptr_right, TS_id_));
    else
      TS_history_.emplace(t_new_TS, TimeSurfaceObservation(cv_ptr_left, cv_ptr_right, TS_id_, true));
    TS_id_++;

    // keep TS_history's size constant
    while (TS_history_.size() > TS_HISTORY_LENGTH_)
    {
      auto it = TS_history_.begin();
      TS_history_.erase(it);
    }
  }

  void esvo_Mapping::reset()
  {
    // mutual-thread communication with MappingThread.
    LOG(INFO) << "Coming into reset()";
    reset_promise_.set_value();
    LOG(INFO) << "(reset) The mapping thread future is waiting for the value.";
    mapping_thread_future_.get();
    LOG(INFO) << "(reset) The mapping thread future receives the value.";

    // clear all maintained data
    events_left_.clear();
    events_right_.clear();
    TS_history_.clear();
    tf_->clear();
    pc_->clear();
    pc_near_->clear();
    pc_global_->clear();
    TS_id_ = 0;
    depthFramePtr_->clear();
    dqvDepthPoints_.clear();

    ebm_.resetParameters(BM_patch_size_X_, BM_patch_size_Y_,
                         BM_min_disparity_, BM_max_disparity_, BM_step_, BM_ZNCC_Threshold_, BM_bUpDownConfiguration_);

    for (int i = 0; i < 2; i++)
      LOG(INFO) << "****************************************************";
    LOG(INFO) << "****************** RESET THE SYSTEM *********************";
    for (int i = 0; i < 2; i++)
      LOG(INFO) << "****************************************************\n\n";

    // restart the mapping thread
    reset_promise_ = std::promise<void>();
    mapping_thread_promise_ = std::promise<void>();
    reset_future_ = reset_promise_.get_future();
    mapping_thread_future_ = mapping_thread_promise_.get_future();
    ESVO_System_Status_ = "INITIALIZATION";
    nh_.setParam("/ESVO_SYSTEM_STATUS", ESVO_System_Status_);
    std::thread MappingThread(&esvo_Mapping::Process, this,
                              std::move(mapping_thread_promise_), std::move(reset_future_));
    MappingThread.detach();
  }

  void esvo_Mapping::onlineParameterChangeCallback(DVS_MappingStereoConfig &config, uint32_t level)
  {
    bool online_parameters_changed = false;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);

      if (invDepth_min_range_ != config.invDepth_min_range ||
          invDepth_max_range_ != config.invDepth_max_range ||
          residual_vis_threshold_ != config.residual_vis_threshold ||
          stdVar_vis_threshold_ != config.stdVar_vis_threshold ||
          age_max_range_ != config.age_max_range ||
          age_vis_threshold_ != config.age_vis_threshold ||
          fusion_radius_ != config.fusion_radius ||
          maxNumFusionFrames_ != config.maxNumFusionFrames ||
          bDenoising_ != config.Denoising ||
          bRegularization_ != config.Regularization ||
          resetButton_ != config.ResetButton ||
          PROCESS_EVENT_NUM_ != config.PROCESS_EVENT_NUM ||
          TS_HISTORY_LENGTH_ != config.TS_HISTORY_LENGTH ||
          BM_min_disparity_ != config.BM_min_disparity ||
          BM_max_disparity_ != config.BM_max_disparity ||
          BM_step_ != config.BM_step ||
          BM_ZNCC_Threshold_ != config.BM_ZNCC_Threshold)
      {
        online_parameters_changed = true;
      }

      invDepth_min_range_ = config.invDepth_min_range;
      invDepth_max_range_ = config.invDepth_max_range;
      residual_vis_threshold_ = config.residual_vis_threshold;
      cost_vis_threshold_ = patch_area_ * pow(residual_vis_threshold_, 2);
      stdVar_vis_threshold_ = config.stdVar_vis_threshold;
      age_max_range_ = config.age_max_range;
      age_vis_threshold_ = config.age_vis_threshold;
      fusion_radius_ = config.fusion_radius;
      maxNumFusionFrames_ = config.maxNumFusionFrames;
      bDenoising_ = config.Denoising;
      bRegularization_ = config.Regularization;
      resetButton_ = config.ResetButton;
      PROCESS_EVENT_NUM_ = config.PROCESS_EVENT_NUM;
      TS_HISTORY_LENGTH_ = config.TS_HISTORY_LENGTH;
      BM_min_disparity_ = config.BM_min_disparity;
      BM_max_disparity_ = config.BM_max_disparity;
      BM_step_ = config.BM_step;
      BM_ZNCC_Threshold_ = config.BM_ZNCC_Threshold;
    }

    if (config.mapping_rate_hz != mapping_rate_hz_)
    {
      changed_frame_rate_ = true;
      online_parameters_changed = true;
      mapping_rate_hz_ = config.mapping_rate_hz;
    }

    if (online_parameters_changed)
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      LOG(INFO) << "onlineParameterChangeCallback ==============";
      reset();
    }
  }

  void esvo_Mapping::publishMappingResults(
      DepthMap::Ptr depthMapPtr,
      Transformation tr,
      ros::Time t)
  {
    cv::Mat invDepthImage, stdVarImage, ageImage, costImage, eventImage, confidenceMap;

    // invDepthImage
    visualizor_.plot_map(depthMapPtr, tools::InvDepthMap, invDepthImage,
                         invDepth_max_range_, invDepth_min_range_, stdVar_vis_threshold_, age_vis_threshold_);
    publishImage(invDepthImage, t, invDepthMap_pub_);

    // stdVarImage
    visualizor_.plot_map(depthMapPtr, tools::StdVarMap, stdVarImage,
                         stdVar_vis_threshold_, 0.0, stdVar_vis_threshold_);
    publishImage(stdVarImage, t, stdVarMap_pub_);

    // ageImage
    visualizor_.plot_map(depthMapPtr, tools::AgeMap, ageImage, age_max_range_, 0, age_vis_threshold_);
    publishImage(ageImage, t, ageMap_pub_);

    // costImage
    visualizor_.plot_map(depthMapPtr, tools::CostMap, costImage, cost_vis_threshold_, 0.0, cost_vis_threshold_);
    publishImage(costImage, t, costMap_pub_);

    if (ESVO_System_Status_ == "INITIALIZATION")
      publishPointCloud(depthMapPtr, tr, t);

    if (ESVO_System_Status_ == "WORKING")
    {
      if (FusionStrategy_ == "CONST_FRAMES")
      {
        if (dqvDepthPoints_.size() == maxNumFusionFrames_)
          publishPointCloud(depthMapPtr, tr, t);
      }
      if (FusionStrategy_ == "CONST_POINTS")
      {
        size_t numFusionPoints = 0;
        for (size_t n = 0; n < dqvDepthPoints_.size(); n++)
          numFusionPoints += dqvDepthPoints_[n].size();
        if (numFusionPoints > 0.5 * maxNumFusionPoints_)
          publishPointCloud(depthMapPtr, tr, t);
      }
    }
  }

  void esvo_Mapping::publishPointCloud(
      DepthMap::Ptr &depthMapPtr,
      Transformation &tr,
      ros::Time &t)
  {
    sensor_msgs::PointCloud2::Ptr pc_to_publish(new sensor_msgs::PointCloud2);
    Eigen::Matrix<double, 4, 4> T_world_result = tr.getTransformationMatrix();

    pc_->clear();
    pc_->reserve(50000);
    pc_near_->clear();
    pc_near_->reserve(50000);

    double FarthestDistance = 0.0;
    Eigen::Vector3d FarthestPoint;

    for (auto it = depthMapPtr->begin(); it != depthMapPtr->end(); it++)
    {
      Eigen::Vector3d p_world = T_world_result.block<3, 3>(0, 0) * it->p_cam() + T_world_result.block<3, 1>(0, 3);
      pc_->push_back(pcl::PointXYZ(p_world(0), p_world(1), p_world(2)));

      if (it->p_cam().norm() < visualize_range_)
        pc_near_->push_back(pcl::PointXYZ(p_world(0), p_world(1), p_world(2)));
      if (it->p_cam().norm() > FarthestDistance)
      {
        FarthestDistance = it->p_cam().norm();
        FarthestPoint = it->p_cam();
      }
    }
#ifdef ESVO_CORE_MAPPING_DEBUG
    LOG(INFO) << "The farthest point (p_cam): " << FarthestPoint.transpose();
#endif

    if (!pc_->empty())
    {
#ifdef ESVO_CORE_MAPPING_DEBUG
      LOG(INFO) << "<<<<<<<<<(pointcloud)<<<<<<<<" << pc_->size() << " points are published";
#endif
      pcl::toROSMsg(*pc_, *pc_to_publish);
      pc_to_publish->header.stamp = t;
      pc_pub_.publish(pc_to_publish);
    }

    // publish global pointcloud
    if (bVisualizeGlobalPC_)
    {
      if (t.toSec() - t_last_pub_pc_ > visualizeGPC_interval_)
      {
        PointCloud::Ptr pc_filtered(new PointCloud());
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(pc_near_);
        sor.setLeafSize(0.03, 0.03, 0.03);
        sor.filter(*pc_filtered);

        // copy the most current pc tp pc_global
        size_t pc_length = pc_filtered->size();
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

  void esvo_Mapping::publishEventFrame(const ros::Time &t)
  {
    cv::Mat eventMap;
    visualizor_.plot_eventMap(vALLEventsPtr_left_,
                              eventMap,
                              camSysPtr_->cam_left_ptr_->height_,
                              camSysPtr_->cam_left_ptr_->width_);
    publishImage(eventMap, t, eventFrame_pub_, "mono8");
  }

  void
  esvo_Mapping::publishImage(
      const cv::Mat &image,
      const ros::Time &t,
      image_transport::Publisher &pub,
      std::string encoding)
  {
    if (pub.getNumSubscribers() == 0)
    {
      //    LOG(INFO) << "------------------------------: " << pub.getTopic();
      return;
    }
    //  LOG(INFO) << "+++++++++++++++++++++++++++++++: " << pub.getTopic();

    std_msgs::Header header;
    header.stamp = t;
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, encoding.c_str(), image).toImageMsg();
    pub.publish(msg);
  }

  void esvo_Mapping::createEdgeMask(
      std::vector<dvs_msgs::Event *> &vEventsPtr,
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
          {
          }
          else
          {
            edgeMap.at<uchar>(y, x) = 255;
            vEdgeletCoordinates.emplace_back((size_t)x, (size_t)y);
          }
        }
      it_tmp++;
    }
  }

  void esvo_Mapping::createDenoisingMask(
      std::vector<dvs_msgs::Event *> &vAllEventsPtr,
      cv::Mat &mask,
      size_t row, size_t col)
  {
    cv::Mat eventMap;
    visualizor_.plot_eventMap(vAllEventsPtr, eventMap, row, col);
    cv::medianBlur(eventMap, mask, 3);
  }

  void esvo_Mapping::extractDenoisedEvents(
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

  /************ publish results *******************/
  void esvo_Mapping::publishPose(const ros::Time &t, Transformation &tr)
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

  void esvo_Mapping::publishPath(const ros::Time &t, Transformation &tr)
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

  void esvo_Mapping::saveTrajectory(const std::string &resultDir)
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