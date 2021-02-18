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

//#define ESVO_CORE_MVSTEREO_LOG

namespace esvo_core
{
  esvo_MVSMono::esvo_MVSMono(
      const ros::NodeHandle &nh,
      const ros::NodeHandle &nh_private)
      : nh_(nh),
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
        invDepth_min_range_(tools::param(pnh_, "invDepth_min_range", 0.16)),
        invDepth_max_range_(tools::param(pnh_, "invDepth_max_range", 2.0)),
        dpSolver_(camSysPtr_, dpConfigPtr_, NUMERICAL, NUM_THREAD_MAPPING),
        dFusor_(camSysPtr_, dpConfigPtr_),
        dRegularizor_(dpConfigPtr_),
        em_(camSysPtr_, NUM_THREAD_MAPPING),
        ebm_(camSysPtr_, NUM_THREAD_MAPPING, tools::param(pnh_, "SmoothTimeSurface", false)),
        pc_(new PointCloud()),
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

    /**** online parameters ***/
    // mapping parameters
    patch_area_ = tools::param(pnh_, "patch_size_X", 25) * tools::param(pnh_, "patch_size_Y", 25);
    residual_vis_threshold_ = tools::param(pnh_, "residual_vis_threshold", 15);
    cost_vis_threshold_ = pow(residual_vis_threshold_, 2) * patch_area_;
    stdVar_vis_threshold_ = tools::param(pnh_, "stdVar_vis_threshold", 0.005);
    age_max_range_ = tools::param(pnh_, "age_max_range", 5);
    age_vis_threshold_ = tools::param(pnh_, "age_vis_threshold", 0);
    fusion_radius_ = tools::param(pnh_, "fusion_radius", 0);
    FusionStrategy_ = tools::param(pnh_, "FUSION_STRATEGY", std::string("CONST_FRAMES"));
    maxNumFusionFrames_ = tools::param(pnh_, "maxNumFusionFrames", 20);
    maxNumFusionPoints_ = tools::param(pnh_, "maxNumFusionPoints", 5000);

    // options
    bDenoising_ = tools::param(pnh_, "Denoising", false);
    bRegularization_ = tools::param(pnh_, "Regularization", false);
    resetButton_ = tools::param(pnh_, "ResetButton", false);

    // module parameters
    PROCESS_EVENT_NUM_ = tools::param(pnh_, "PROCESS_EVENT_NUM", 500);
    TS_HISTORY_LENGTH_ = tools::param(pnh_, "TS_HISTORY_LENGTH", 100);
    mapping_rate_hz_ = tools::param(pnh_, "mapping_rate_hz", 20);

    // EM parameters [26]
    EM_Slice_Thickness_ = tools::param(pnh_, "EM_Slice_Thickness", 1e-3);
    EM_Time_THRESHOLD_ = tools::param(pnh_, "EM_Time_THRESHOLD", 5e-5);
    EM_EPIPOLAR_THRESHOLD_ = tools::param(pnh_, "EM_EPIPOLAR_THRESHOLD", 0.5);
    EM_TS_NCC_THRESHOLD_ = tools::param(pnh_, "EM_TS_NCC_THRESHOLD", 0.1);
    EM_patch_size_X_ = tools::param(pnh_, "patch_size_X", 25);
    EM_patch_size_Y_ = tools::param(pnh_, "patch_size_Y", 25);
    EM_numEventMatching_ = tools::param(pnh_, "EM_NUM_EVENT_MATCHING", 3000);
    EM_patch_intensity_threshold_ = tools::param(pnh_, "EM_PATCH_INTENSITY_THRESHOLD", 125);
    EM_patch_valid_ratio_ = tools::param(pnh_, "EM_PATCH_VALID_RATIO", 0.1);
    em_.resetParameters(EM_Time_THRESHOLD_, EM_EPIPOLAR_THRESHOLD_, EM_TS_NCC_THRESHOLD_,
                        EM_patch_size_X_, EM_patch_size_Y_, EM_patch_intensity_threshold_, EM_patch_valid_ratio_);
    // Event Block Matching (BM) parameters
    BM_half_slice_thickness_ = tools::param(pnh_, "BM_half_slice_thickness", 0.001);
    BM_MAX_NUM_EVENTS_PER_MATCHING_ = tools::param(pnh_, "BM_MAX_NUM_EVENTS_PER_MATCHING", 300);
    BM_patch_size_X_ = tools::param(pnh_, "patch_size_X", 25);
    BM_patch_size_Y_ = tools::param(pnh_, "patch_size_Y", 25);
    BM_min_disparity_ = tools::param(pnh_, "BM_min_disparity", 3);
    BM_max_disparity_ = tools::param(pnh_, "BM_max_disparity", 40);
    BM_step_ = tools::param(pnh_, "BM_step", 1);
    BM_ZNCC_Threshold_ = tools::param(pnh_, "BM_ZNCC_Threshold", 0.1);
    BM_bUpDownConfiguration_ = tools::param(pnh_, "BM_bUpDownConfiguration", false);

    // SGM [45] parameters
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

#ifdef ESVO_CORE_MVSTEREO_LOG
    LOG(INFO) << "f: " << f << " "
              << " b: " << b;
    LOG(INFO) << "invDepth_min_range_: " << invDepth_min_range_;
    LOG(INFO) << "invDepth_max_range_: " << invDepth_max_range_;
    LOG(INFO) << "minDisparity: " << minDisparity;
    LOG(INFO) << "maxDisparity: " << maxDisparity;
#endif

    // mvstereo mode
    size_t MVStereoMode = tools::param(pnh_, "MVStereoMode", 1);
    msm_ = (eMVStereoMode)MVStereoMode;

    // initialize Event Block Matcher
    ebm_.resetParameters(BM_patch_size_X_, BM_patch_size_Y_, minDisparity, maxDisparity,
                         BM_step_, BM_ZNCC_Threshold_, BM_bUpDownConfiguration_);

    // callbacks functions
    events_left_sub_ = nh_.subscribe<dvs_msgs::EventArray>("events_left", 0, boost::bind(&esvo_MVSMono::eventsCallback, this, _1, boost::ref(events_left_)));
    events_right_sub_ = nh_.subscribe<dvs_msgs::EventArray>("events_right", 0, boost::bind(&esvo_MVSMono::eventsCallback, this, _1, boost::ref(events_right_)));
    stampedPose_sub_ = nh_.subscribe("stamped_pose", 0, &esvo_MVSMono::stampedPoseCallback, this);
    TS_sync_.registerCallback(boost::bind(&esvo_MVSMono::timeSurfaceCallback, this, _1, _2));
    // TF
    tf_ = std::make_shared<tf::Transformer>(true, ros::Duration(100.0));

    // result publishers
    invDepthMap_pub_ = it_.advertise("Inverse_Depth_Map", 1);
    stdVarMap_pub_ = it_.advertise("Standard_Variance_Map", 1);
    ageMap_pub_ = it_.advertise("Age_Map", 1);
    costMap_pub_ = it_.advertise("Cost_Map", 1);
    pc_pub_ = nh_.advertise<PointCloud>("/esvo_mvsmono/pointcloud_world", 1);
    pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/esvo_mvsmono/pose_pub", 1);

    TotalNumFusion_ = 0;

    // multi-thread management
    mapping_thread_future_ = mapping_thread_promise_.get_future();
    reset_future_ = reset_promise_.get_future();

    // stereo mapping detached thread
    std::thread MappingThread(&esvo_MVSMono::MappingLoop, this,
                              std::move(mapping_thread_promise_), std::move(reset_future_));
    MappingThread.detach();

    // Dynamic reconfigure
    dynamic_reconfigure_callback_ = boost::bind(&esvo_MVSMono::onlineParameterChangeCallback, this, _1, _2);
    server_.reset(new dynamic_reconfigure::Server<DVS_MappingStereoConfig>(nh_private));
    server_->setCallback(dynamic_reconfigure_callback_);

    resultPath_ = tools::param(pnh_, "PATH_TO_SAVE_RESULT", std::string());
    KEYFRAME_LINEAR_DIS_ = tools::param(pnh_, "KEYFRAME_LINEAR_DIS", 0.2);
    KEYFRAME_ORIENTATION_DIS_ = tools::param(pnh_, "KEYFRAME_ORIENTATION_DIS", 5); // deg
    KEYFRAME_MEANDEPTH_DIS_ = tools::param(pnh_, "KEYFRAME_MEANDEPTH_DIS", 0.15);     // percentage

    // DSI_Mapper configure
    emvs_dsi_shape_.printDSIInfo();

    isKeyframe_ = false;
    T_w_frame_.setIdentity();
    T_w_keyframe_.setIdentity();
    meanDepth_ = -1.0;
    mAllPoses_.clear();

    depthMap_pub_ = it_.advertise("DSI_Depth_Map", 1);
    confidenceMap_pub_ = it_.advertise("DSI_Confidence_Map", 1);
    semiDenseMask_pub_ = it_.advertise("DSI_Semi_Dense_Mask", 1);
    varianceMap_pub_ = it_.advertise("DSI_Variance_Map", 1);
    EMVS_Init_event_ = tools::param(pnh_, "EMVS_Init_event", 5e4);
    EMVS_Keyframe_event_ = tools::param(pnh_, "EMVS_Accu_event", 2e5); // it should be: EMVS_Keyframe_event_ > EMVS_Init_event_
    SAVE_RESULT_ = tools::param(pnh_, "SAVE_RESULT", false);
    strDataset_ = tools::param(pnh_, "Dataset_Name", std::string("rpg_stereo"));
    T_world_map_.setIdentity();
  }

  esvo_MVSMono::~esvo_MVSMono()
  {
    pc_pub_.shutdown();
    pose_pub_.shutdown();
    invDepthMap_pub_.shutdown();
    stdVarMap_pub_.shutdown();
    ageMap_pub_.shutdown();
    costMap_pub_.shutdown();
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
#ifdef ESVO_CORE_MVSTEREO_LOG
      ROS_INFO("Changing mapping framerate to %d Hz", mapping_rate_hz_);
#endif
      r = ros::Rate(mapping_rate_hz_);
      changed_frame_rate_ = false;
    }
    //
    if(TS_history_.size() >= 10)/* To assure the esvo_time_surface node has been working. */
    {

#ifdef ESVO_CORE_MVSTEREO_LOG
      LOG(INFO) << "TS_history_.size(): " << TS_history_.size();
#endif
      while(true)
      {
        if(data_mutex_.try_lock())
        {
          dataTransferring();
          data_mutex_.unlock();
          break;
        }
        else
        {
          if(future_reset.wait_for(std::chrono::nanoseconds(1)) == std::future_status::ready)
          {
            prom_mapping.set_value();
            return;
          }
        }
      }

      // To check if the most current TS observation has been loaded by dataTransferring()
      if(TS_obs_.second.isEmpty())
      {
        r.sleep();
        continue;
      }
      publishKFPose(TS_obs_.first, TS_obs_.second.tr_);
      if (msm_ == PURE_EMVS || msm_ == PURE_EMVS_PLUS_ESTIMATION)
      {
        insertKeyframe();
      }
      MappingAtTime(TS_obs_.first);
    }
    else
    {
      if(future_reset.wait_for(std::chrono::nanoseconds(1)) == std::future_status::ready)
      {
        prom_mapping.set_value();
        return;
      }
    }
    r.sleep();
  }
}

void esvo_MVSMono::MappingAtTime(const ros::Time& t)
{
  TicToc tt_mapping;
  double t_overall_count = 0;

  /**************************************************/
  /*********** 0. set the current DepthFrame ********/
  /**************************************************/
  DepthFrame::Ptr depthFramePtr_new = std::make_shared<DepthFrame>(
    camSysPtr_->cam_left_ptr_->height_, camSysPtr_->cam_left_ptr_->width_);
  depthFramePtr_new->setId(TS_obs_.second.id_);
  depthFramePtr_new->setTransformation(TS_obs_.second.tr_);
  depthFramePtr_ = depthFramePtr_new;

  /******************************************************************/
  /*************** Event Multi-Stereo Mapping (EMVS) ****************/
  /******************************************************************/
  // DSM initialization entrance: https://github.com/jzubizarreta/dsm/blob/master/dsm/src/FullSystem/FullSystem.cpp#L419
  // DSM initialization: https://github.com/jzubizarreta/dsm/blob/f84ead96546bd16291b107dd513aedffbcb0bd05/dsm/src/Initializer/MonoInitializer.cpp#L102
  if (msm_ == PURE_EMVS || msm_ == PURE_EMVS_PLUS_ESTIMATION)
  {
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
    }

    // undistort events' coordinates
    std::vector<Eigen::Vector4d> vEdgeletCoordinates;
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
    // LOG(INFO) << "Process " << vEdgeletCoordinates.size() << " events";
    // LOG(INFO) << mVirtualPoses_.front().first.toSec() << " " << vEdgeletCoordinates.front()[2];
    // LOG(INFO) << mVirtualPoses_.back().first.toSec() << " " << vEdgeletCoordinates.back()[2];

    double t_solve, t_fusion, t_optimization, t_regularization;
    tt_mapping.tic();
    if (isKeyframe_)
    {
      LOG(INFO) << "insert a keyframe: reset the DSI for the local map\r";
      if (emvs_mapper_.accu_event_number_ <= EMVS_Keyframe_event_)
        if (!dqvDepthPoints_.empty())
          dqvDepthPoints_.pop_back();
      dqvDepthPoints_.push_back(std::vector<DepthPoint>());
      emvs_mapper_.dsi_.writeGridNpy(std::string(resultPath_ + "dsi.npy").c_str());
      emvs_mapper_.reset();
    }
    else
    {
      // LOG(INFO) << "insert an non-keyframe: add events onto the DSI\r";
    }

    emvs_mapper_.storeEventsPose(mVirtualPoses_, vEdgeletCoordinates);
    if (!emvs_mapper_.dsiInitFlag_)
    {
      TS_obs_.second.getTimeSurfaceNegative(0);
      TS_map_negative_history_.emplace(t, std::make_shared<Eigen::MatrixXd>(TS_obs_.second.TS_negative_left_));
      if (emvs_mapper_.storeEventNum() > EMVS_Init_event_)
      {
        Eigen::Matrix4d T_w_rv;
        ros::Time t_rv = emvs_mapper_.getRVTime();
        trajectory_.getPoseAt(mAllPoses_, t_rv, T_w_rv);
        std::map<ros::Time, std::shared_ptr<Eigen::MatrixXd>>::iterator it_TS_negative =
            std::upper_bound(TS_map_negative_history_.begin(), TS_map_negative_history_.end(), t_rv,
                            [](const ros::Time &t, const std::pair<ros::Time, std::shared_ptr<Eigen::MatrixXd>> &tso) {
                              return t.toSec() < tso.first.toSec();
                            });
        emvs_mapper_.initializeDSI(T_w_rv);
        // emvs_mapper_.computeObservation(2e3);
        // emvs_mapper_.setTSNegativeObservation(it_TS_negative->second);
        TS_map_negative_history_.clear();
        // LOG(INFO) << "t_rv: " << t_rv << ", " << "it_TS_negative: " << it_TS_negative->first;
      }
    }
    if (emvs_mapper_.dsiInitFlag_)
    {
      emvs_mapper_.updateDSI();
      emvs_mapper_.clearEvents();
    }
    cv::Mat depth_map, confidence_map, semidense_mask;
    emvs_mapper_.getDepthMapFromDSI(depth_map, confidence_map, semidense_mask, emvs_opts_depth_map_, meanDepth_);
    std::vector<DepthPoint> &vdp = dqvDepthPoints_.back(); // depth points on the current observations
    emvs_mapper_.getDepthPoint(depth_map, confidence_map, semidense_mask, vdp);
    t_solve = tt_mapping.toc();
    LOG_EVERY_N(INFO, 50) << "Get DP from DSI costs: " << t_solve << " ms\r"; // 40ms

    if (msm_ == PURE_EMVS)
    {
      if (FusionStrategy_ == "CONST_POINTS")
      {
        size_t numFusionPoints = 0;
        tt_mapping.tic();
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
        while (dqvDepthPoints_.size() > maxNumFusionFrames_)
          dqvDepthPoints_.pop_front();
      }
      else
      {
        LOG(INFO) << "Invalid FusionStrategy is assigned.";
        exit(-1);
      }
      for (auto it = dqvDepthPoints_.rbegin(); it != dqvDepthPoints_.rend(); it++)
        dFusor_.naive_propagation(*it, depthFramePtr_);

      // regularization
      if (bRegularization_)
      {
        tt_mapping.tic();
        dRegularizor_.apply(depthFramePtr_->dMap_);
        t_regularization = tt_mapping.toc();
      }
    }
    /**************************************************************/
    /*************  Nonlinear Optimization & Fusion ***************/
    /**************************************************************/
    // TODO: since the depth point is not properly modeled as Guassian distribution
    else 
    {
      if (FusionStrategy_ == "CONST_POINTS")
      {
        size_t numFusionPoints = 0;
        tt_mapping.tic();
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
        while (dqvDepthPoints_.size() > maxNumFusionFrames_)
          dqvDepthPoints_.pop_front();
      }
      else
      {
        LOG(INFO) << "Invalid FusionStrategy is assigned.";
        exit(-1);
      }

      // apply fusion and count the total number of fusion.
      size_t numFusionCount = 0;
      for (auto it = dqvDepthPoints_.rbegin(); it != dqvDepthPoints_.rend(); it++)
      {
        numFusionCount += dFusor_.update(*it, depthFramePtr_, fusion_radius_);
        //    LOG(INFO) << "numFusionCount: " << numFusionCount;
      }

      TotalNumFusion_ += numFusionCount;
      depthFramePtr_->dMap_->clean(
          pow(stdVar_vis_threshold_, 2), age_vis_threshold_, invDepth_max_range_, invDepth_min_range_);
      t_fusion = tt_mapping.toc();

      // regularization
      if (bRegularization_)
      {
        tt_mapping.tic();
        dRegularizor_.apply(depthFramePtr_->dMap_);
        t_regularization = tt_mapping.toc();
      }
      t_optimization = t_solve + t_fusion + t_regularization;
      t_overall_count += t_optimization;
    }

    std::thread tPublishDSIResult(&esvo_MVSMono::publishDSIResults, this,
                                  t, semidense_mask, depth_map, confidence_map);
    tPublishDSIResult.detach();

    // visualization
    std::thread tPublishMappingResult(&esvo_MVSMono::publishMappingResults, this,
                                      depthFramePtr_->dMap_, depthFramePtr_->T_world_frame_, t);
    tPublishMappingResult.detach();
    // LOG_EVERY_N(INFO, 20) << "Depth point size: " << depthFramePtr_->dMap_->size(); // 4000

    // To save the depth result, set it to true.
    if (SAVE_RESULT_)
    {
      // For quantitative comparison.
      std::string baseDir(resultPath_);
      std::string saveDir(baseDir);
      saveDepthMap(depthFramePtr_->dMap_, saveDir, t);
    }
    return;
  }

#ifdef  ESVO_CORE_MVSTEREO_LOG
  LOG(INFO) << "\n";
  LOG(INFO) << "------------------------------------------------------------";
  LOG(INFO) << "--------------------Computation Cost-------------------------";
  LOG(INFO) << "------------------------------------------------------------";
  LOG(INFO) << "Denoising: " << t_denoising << " ms, (" << t_denoising / t_overall_count * 100 << "%).";
  LOG(INFO) << "------------------------------------------------------------";
  LOG(INFO) << "------------------------------------------------------------";
  LOG(INFO) << "Update: " << t_optimization << " ms, (" << t_optimization / t_overall_count * 100
            << "%).";
  LOG(INFO) << "-- nonlinear optimization: " << t_solve << " ms, (" << t_solve / t_overall_count * 100
            << "%).";
  LOG(INFO) << "-- fusion (" << TotalNumFusion_ << "): " << t_fusion << " ms, (" << t_fusion / t_overall_count * 100
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

/**
 * @brief: This function defines xx criterias to indicate if insert new keyframe in Mapping
 */
void esvo_MVSMono::insertKeyframe()
{
  // criterion in LSD-SLAM or EVO
  // This is done based on two weights, the relative distance to the current key-frame and
  // the angle to the current key-frame. Is the weighted sum of these two larger than a certain threshold,
  // a new key-frame is taken.
  T_w_frame_ = TS_obs_.second.tr_.getTransformationMatrix();
  double dis = (T_w_frame_.topRightCorner<3, 1>() - T_w_keyframe_.topRightCorner<3, 1>()).norm();
  // if (dis > KEYFRAME_LINEAR_DIS_)
  if (meanDepth_ < 0 || (meanDepth_ != 0 && dis > KEYFRAME_MEANDEPTH_DIS_ * meanDepth_)) // 1e5: system just starts; > 0.15: move at a long distance
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

bool esvo_MVSMono::dataTransferring()
{
  TS_obs_ = std::make_pair(ros::Time(), TimeSurfaceObservation());// clean the TS obs.
  if(TS_history_.size() <= 10)
    return false;
  totalNumCount_ = 0;

  // load current Time-Surface Observation
  auto it_end = TS_history_.rbegin();
  it_end++;// in case that the tf is behind the most current TS.
  auto it_begin = TS_history_.begin();
  while(TS_obs_.second.isEmpty())
  {
    Transformation tr;
    if(getPoseAt(it_end->first, tr, dvs_frame_id_))
    {
      it_end->second.setTransformation(tr);
      TS_obs_ = *it_end;
    }
    if(it_end->first == it_begin->first)
      break;
    it_end++;
  }
  if(TS_obs_.second.isEmpty())
    return false;

  // EMVS
  if (msm_ == PURE_EMVS || msm_ == PURE_EMVS_PLUS_ESTIMATION)
  {
    // copy all involved events' pointers
    vALLEventsPtr_left_.clear();   // to generate denoising mask (only used for ECCV 2018 dataset)
    vCloseEventsPtr_left_.clear(); // will be denoised using the mask above.

    // load allEvent
    ros::Time t_end = TS_obs_.first;
    ros::Time t_begin(std::max(0.0, t_end.toSec() - 10 * BM_half_slice_thickness_));
    auto ev_end_it = tools::EventBuffer_lower_bound(events_left_, t_end);
    auto ev_begin_it = tools::EventBuffer_lower_bound(events_left_, t_begin); //events_left_.begin();
    const size_t MAX_NUM_Event_INVOLVED = 20000;
    vALLEventsPtr_left_.reserve(MAX_NUM_Event_INVOLVED);
    vCloseEventsPtr_left_.reserve(MAX_NUM_Event_INVOLVED);
    while (ev_end_it != ev_begin_it && vALLEventsPtr_left_.size() < MAX_NUM_Event_INVOLVED)
    {
      vALLEventsPtr_left_.push_back(ev_end_it._M_cur);
      vCloseEventsPtr_left_.push_back(ev_end_it._M_cur);
      ev_end_it--;
    }
    totalNumCount_ = vCloseEventsPtr_left_.size();
    // LOG(INFO) << "receive " << totalNumCount_ << " events";

    // using inherent linear interpolation
    // t0 <= ev.ts <= t1
    // pose_0 (t0), pose_1 (t1), ..., pose_N (tN)
    // t_begin, t_tmp, ..., t_end
    mVirtualPoses_.clear();
    ros::Time t_tmp = t_begin;
    while (t_tmp.toSec() <= t_end.toSec())
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
      t_tmp = ros::Time(t_tmp.toSec() + 0.05 * BM_half_slice_thickness_);
    }
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
    tf_->getLatestCommonTime(dvs_frame_id_.c_str(), ps_msg->header.frame_id, tf_lastest_common_time_, err_tf);
  delete err_tf;

  if( tf_lastest_common_time_.toSec() != 0)
  {
    const double dt = stamp_first_event.toSec() - tf_lastest_common_time_.toSec();
    if(dt < 0 || std::fabs(dt) >= max_time_diff_before_reset_s)
    {
      ROS_INFO("Inconsistent event timestamps detected <stampedPoseCallback> (new: %f, old %f), resetting.",
               stamp_first_event.toSec(), tf_lastest_common_time_.toSec());
      reset();
    }
  }

  // add pose to tf
  // tf::Transform tf(
  //     tf::Quaternion(
  //         ps_msg->pose.orientation.x,
  //         ps_msg->pose.orientation.y,
  //         ps_msg->pose.orientation.z,
  //         ps_msg->pose.orientation.w),
  //     tf::Vector3(
  //         ps_msg->pose.position.x,
  //         ps_msg->pose.position.y,
  //         ps_msg->pose.position.z));
  // tf::StampedTransform st(tf, ps_msg->header.stamp, ps_msg->header.frame_id, dvs_frame_id_.c_str());
  // tf_->setTransform(st);

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
  else if (!strDataset_.compare("upen"))
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
  tf::StampedTransform st(tf, ps_msg->header.stamp, ps_msg->header.frame_id, dvs_frame_id_.c_str());
  tf_->setTransform(st);

  static tf::TransformBroadcaster br;
  br.sendTransform(st);

  if (msm_ == PURE_EMVS || msm_ == PURE_EMVS_PLUS_ESTIMATION)
  {
    // add pose to mAllPoses_
    mAllPoses_.emplace(ps_msg->header.stamp, T_map_cam);
  }
}

// return the pose of the left event cam at time t.
bool esvo_MVSMono::getPoseAt(const ros::Time &t,
  esvo_core::Transformation &Tr,// T_world_virtual
  const std::string& source_frame )
{
  std::string* err_msg = new std::string();
  if(!tf_->canTransform(world_frame_id_, source_frame, t, err_msg))
  {
#ifdef ESVO_CORE_MVSTEREO_LOG
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

    // HARDCODED: The GT pose of rpg dataset is the pose of stereo rig, namely that of the marker.
    if(std::strcmp(source_frame.c_str(), "marker") == 0)
    {
      Transformation ::TransformationMatrix T_marker_cam;
      T_marker_cam << 5.363262328777285e-01, -1.748374625145743e-02, -8.438296573030597e-01, -7.009849865398374e-02,
          8.433577587813513e-01, -2.821937531845164e-02, 5.366109927684415e-01, 1.881333563905305e-02,
          -3.319431623758162e-02, -9.994488408486204e-01, -3.897382049768972e-04, -6.966829200678797e-02,
          0, 0, 0, 1;
      Transformation::TransformationMatrix T_world_marker =
        Tr.getTransformationMatrix();
      Transformation::TransformationMatrix T_world_cam =
        T_world_marker * T_marker_cam;
      Tr = Transformation(T_world_cam);
    }
    return true;
  }
}

void esvo_MVSMono::eventsCallback(
  const dvs_msgs::EventArray::ConstPtr& msg,
  EventQueue& EQ)
{
  std::lock_guard<std::mutex> lock(data_mutex_);

  static constexpr double max_time_diff_before_reset_s = 0.5;
  const ros::Time stamp_first_event = msg->events[0].ts;

  // check time stamp inconsistency
  if(!msg->events.empty() && !EQ.empty())
  {
    const double dt = stamp_first_event.toSec() - EQ.back().ts.toSec();
    if(dt < 0 || std::fabs(dt) >= max_time_diff_before_reset_s)
    {
      ROS_INFO("Inconsistent event timestamps detected <eventCallback> (new: %f, old %f), resetting.",
               stamp_first_event.toSec(), events_left_.back().ts.toSec());
      reset();
    }
  }

  // add new ones and remove old ones
  for(const dvs_msgs::Event& e : msg->events)
  {
    EQ.push_back(e);
    int i = EQ.size() - 2;
    while(i >= 0 && EQ[i].ts > e.ts) // we may have to sort the queue, just in case the raw event messages do not come in a chronological order.
    {
      EQ[i+1] = EQ[i];
      i--;
    }
    EQ[i+1] = e;
  }
  clearEventQueue(EQ);
}

void esvo_MVSMono::clearEventQueue(EventQueue& EQ)
{
  static constexpr size_t MAX_EVENT_QUEUE_LENGTH = 3000000;
  if (EQ.size() > MAX_EVENT_QUEUE_LENGTH)
  {
    size_t NUM_EVENTS_TO_REMOVE = EQ.size() - MAX_EVENT_QUEUE_LENGTH;
    EQ.erase(EQ.begin(), EQ.begin() + NUM_EVENTS_TO_REMOVE);
  }
}

void esvo_MVSMono::timeSurfaceCallback(
  const sensor_msgs::ImageConstPtr& time_surface_left,
  const sensor_msgs::ImageConstPtr& time_surface_right)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  // check time-stamp inconsistency
  if(!TS_history_.empty())
  {
    static constexpr double max_time_diff_before_reset_s = 0.5;
    const ros::Time stamp_last_image = TS_history_.rbegin()->first;
    const double dt = time_surface_left->header.stamp.toSec() - stamp_last_image.toSec();
    if(dt < 0 || std::fabs(dt) >= max_time_diff_before_reset_s)
    {
      ROS_INFO("Inconsistent frame timestamp detected <timeSurfaceCallback> (new: %f, old %f), resetting.",
               time_surface_left->header.stamp.toSec(), stamp_last_image.toSec());
      reset();
    }
  }

  cv_bridge::CvImagePtr cv_ptr_left, cv_ptr_right;
  try
  {
    cv_ptr_left  = cv_bridge::toCvCopy(time_surface_left, sensor_msgs::image_encodings::MONO8);
    cv_ptr_right = cv_bridge::toCvCopy(time_surface_right, sensor_msgs::image_encodings::MONO8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  // push back the new time surface map
  ros::Time t_new_TS = time_surface_left->header.stamp;
  // Made the gradient computation optional which is up to the jacobian choice.
  if(dpSolver_.getProblemType() == NUMERICAL)
    TS_history_.emplace(t_new_TS, TimeSurfaceObservation(cv_ptr_left, cv_ptr_right, TS_id_));
  else
    TS_history_.emplace(t_new_TS, TimeSurfaceObservation(cv_ptr_left, cv_ptr_right, TS_id_, true));
  TS_id_++;

  // keep TS_history's size constant
  while(TS_history_.size() > TS_HISTORY_LENGTH_)
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
  events_right_.clear();
  TS_history_.clear();
  tf_->clear();
  pc_->clear();
  TS_id_ = 0;
  depthFramePtr_->clear();
  dqvDepthPoints_.clear();

  if(msm_ == PURE_EVENT_MATCHING || msm_ == EM_PLUS_ESTIMATION)
    em_.resetParameters(EM_Time_THRESHOLD_, EM_EPIPOLAR_THRESHOLD_, EM_TS_NCC_THRESHOLD_,
                        EM_patch_size_X_, EM_patch_size_Y_, EM_patch_intensity_threshold_, EM_patch_valid_ratio_);
  if(msm_ == PURE_BLOCK_MATCHING || msm_ == EM_PLUS_ESTIMATION)
    ebm_.resetParameters(BM_patch_size_X_, BM_patch_size_Y_,
                         BM_min_disparity_, BM_max_disparity_, BM_step_, BM_ZNCC_Threshold_, BM_bUpDownConfiguration_);

  for(int i = 0;i < 2;i++)
    LOG(INFO) << "****************************************************";
  LOG(INFO) << "****************** RESET THE SYSTEM *********************";
  for(int i = 0;i < 2;i++)
    LOG(INFO) << "****************************************************\n\n";

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

    if(invDepth_min_range_ != config.invDepth_min_range ||
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
       EM_Time_THRESHOLD_ != config.EM_TIME_THRESHOLD ||
       EM_EPIPOLAR_THRESHOLD_ != config.EM_EPIPOLAR_THRESHOLD ||
       EM_TS_NCC_THRESHOLD_ != config.EM_TS_NCC_THRESHOLD ||
       EM_numEventMatching_ != config.EM_NUM_EVENT_MATCHING ||
       EM_patch_intensity_threshold_ != config.EM_PATCH_INTENSITY_THRESHOLD ||
       EM_patch_valid_ratio_ != config.EM_PATCH_VALID_RATIO ||
       BM_MAX_NUM_EVENTS_PER_MATCHING_ != config.BM_MAX_NUM_EVENTS_PER_MATCHING ||
       BM_min_disparity_ != config.BM_min_disparity ||
       BM_max_disparity_ != config.BM_max_disparity ||
       BM_step_ != config.BM_step ||
       BM_ZNCC_Threshold_ != config.BM_ZNCC_Threshold)
    {
      have_display_parameters_changed = true;
    }

    invDepth_min_range_ = config.invDepth_min_range;
    invDepth_max_range_ = config.invDepth_max_range;
    residual_vis_threshold_ = config.residual_vis_threshold;
    cost_vis_threshold_ = patch_area_ * pow(residual_vis_threshold_,2);
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

    EM_Time_THRESHOLD_ = config.EM_TIME_THRESHOLD;
    EM_EPIPOLAR_THRESHOLD_ = config.EM_EPIPOLAR_THRESHOLD;
    EM_TS_NCC_THRESHOLD_ = config.EM_TS_NCC_THRESHOLD;
    EM_numEventMatching_ = config.EM_NUM_EVENT_MATCHING;
    EM_patch_intensity_threshold_ = config.EM_PATCH_INTENSITY_THRESHOLD;
    EM_patch_valid_ratio_ = config.EM_PATCH_VALID_RATIO;

    BM_MAX_NUM_EVENTS_PER_MATCHING_ = config.BM_MAX_NUM_EVENTS_PER_MATCHING;
    BM_min_disparity_ = config.BM_min_disparity;
    BM_max_disparity_ = config.BM_max_disparity;
    BM_step_ = config.BM_step;
    BM_ZNCC_Threshold_ = config.BM_ZNCC_Threshold;
  }

  if(config.mapping_rate_hz != mapping_rate_hz_)
  {
    changed_frame_rate_ = true;
    have_display_parameters_changed = true;
    mapping_rate_hz_ = config.mapping_rate_hz;
  }

  if(have_display_parameters_changed)
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

void esvo_MVSMono::publishMappingResults(
  DepthMap::Ptr depthMapPtr,
  Transformation tr,
  ros::Time t)
{
  cv::Mat invDepthImage, stdVarImage, ageImage, costImage, eventImage, confidenceMap;

  visualizor_.plot_map(depthMapPtr, tools::InvDepthMap, invDepthImage, invDepth_max_range_, invDepth_min_range_, stdVar_vis_threshold_, age_vis_threshold_);
  publishImage(invDepthImage, t, invDepthMap_pub_);

  visualizor_.plot_map(depthMapPtr, tools::StdVarMap,stdVarImage, stdVar_vis_threshold_, 0.0, stdVar_vis_threshold_);
  publishImage(stdVarImage, t, stdVarMap_pub_);

  visualizor_.plot_map(depthMapPtr, tools::AgeMap,ageImage, age_max_range_, 0, age_vis_threshold_);
  publishImage(ageImage, t, ageMap_pub_);

  visualizor_.plot_map(depthMapPtr, tools::CostMap, costImage, cost_vis_threshold_, 0.0, cost_vis_threshold_);
  publishImage(costImage, t, costMap_pub_);

  publishPointCloud(depthMapPtr, tr, t);
}

void esvo_MVSMono::saveDepthMap(
  DepthMap::Ptr& depthMapPtr,
  std::string& saveDir,
  ros::Time t)
{
  std::ofstream of;
  std::string savePath(saveDir.append(std::to_string(t.toNSec())));
  savePath.append(".txt");
  of.open(savePath, std::ofstream::out);
  if(of.is_open())
  {
    for (auto it = depthMapPtr->begin(); it != depthMapPtr->end(); it++)
    {
      if(it->valid())
        of << it->x().transpose() << " " << it->p_cam()(2) << "\n";
    }
  }
  of.close();
}

void esvo_MVSMono::publishPointCloud(
  DepthMap::Ptr& depthMapPtr,
  Transformation & tr,
  ros::Time& t)
{
  sensor_msgs::PointCloud2::Ptr pc_to_publish (new sensor_msgs::PointCloud2);
  Eigen::Matrix<double, 4, 4> T_world_result = tr.getTransformationMatrix();
  pc_->clear();
  pc_->reserve(50000);
  double FarthestDistance = 0.0;
  Eigen::Vector3d FarthestPoint;

  for(auto it = depthMapPtr->begin();it != depthMapPtr->end();it++)
  {
    Eigen::Vector3d p_world = T_world_result.block<3,3>(0,0) * it->p_cam()
                              + T_world_result.block<3,1>(0,3);
    pc_->push_back(pcl::PointXYZ(p_world(0), p_world(1), p_world(2)));

    if(it->p_cam().norm() > FarthestDistance)
    {
      FarthestDistance = it->p_cam().norm();
      FarthestPoint = it->p_cam();
    }
  }
#ifdef ESVO_CORE_MVSTEREO_LOG
  LOG(INFO) << "The farthest point (p_cam): " << FarthestPoint.transpose();
#endif


  if (!pc_->empty())
  {
#ifdef ESVO_MVSMONO_TRACKING_DEBUG
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::RadiusOutlierRemoval<pcl::PointXYZ> outlier_rm;
  outlier_rm.setInputCloud(pc_);
  outlier_rm.setRadiusSearch(emvs_opts_pc_.radius_search_);
  outlier_rm.setMinNeighborsInRadius(emvs_opts_pc_.min_num_neighbors_);
  outlier_rm.filter(*cloud_filtered);
  pc_->swap(*cloud_filtered);
#endif 

#ifdef ESVO_CORE_MVSTEREO_LOG
    LOG(INFO) << "<<<<<<<<<(pointcloud)<<<<<<<<" << pc_->size() << " points are published";
#endif
    pcl::toROSMsg(*pc_, *pc_to_publish);
    pc_to_publish->header.stamp = t;
    pc_pub_.publish(pc_to_publish);
  }
}

void esvo_MVSMono::publishKFPose(const ros::Time& t, Transformation& tr)
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

void
esvo_MVSMono::publishImage(
  const cv::Mat &image,
  const ros::Time & t,
  image_transport::Publisher & pub,
  std::string encoding)
{
  if(pub.getNumSubscribers() == 0)
    return;

  std_msgs::Header header;
  header.stamp = t;
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, encoding.c_str(), image).toImageMsg();
  pub.publish(msg);
}

void esvo_MVSMono::vEMP2vDP(
  std::vector<EventMatchPair>& vEMP,
  std::vector<DepthPoint>& vdp)
{
  vdp.reserve(vEMP.size());
  double var_pseudo = 0;//pow(stdVar_vis_threshold_*0.99,2);// used for being compatible with the interface, just for visualization.
  for (size_t i = 0; i < vEMP.size(); i++)
  {
    DepthPoint dp(std::floor(vEMP[i].x_left_(1)), std::floor(vEMP[i].x_left_(0)));
    dp.update_x(vEMP[i].x_left_);
    Eigen::Vector3d p_cam;
    camSysPtr_->cam_left_ptr_->cam2World(vEMP[i].x_left_, vEMP[i].invDepth_, p_cam);
    dp.update_p_cam(p_cam);
    dp.update(vEMP[i].invDepth_, var_pseudo);
    dp.residual() = vEMP[i].cost_;
    dp.age() = age_vis_threshold_;
//    LOG(INFO) << "**************esvo_MVSMono::vEMP2vDP: " << dp.variance();
    Eigen::Matrix<double, 4, 4> T_world_cam = vEMP[i].trans_.getTransformationMatrix();
//        LOG(INFO) << "**************esvo_MVSMono::vEMP2vDP: " << T_world_cam;
    dp.updatePose(T_world_cam);
    vdp.push_back(dp);
  }
}

void esvo_MVSMono::eventSlicingForEM(std::vector<EventSlice>& eventSlices)
{
  size_t numSlice = std::floor(
    (t_upBound_.toSec() - t_lowBound_.toSec()) / EM_Slice_Thickness_);// a small number of events are ignored at this step.
  eventSlices.reserve(numSlice);
  std::vector<dvs_msgs::Event *>::iterator it_tmp = vEventsPtr_left_.begin();
  size_t totalNumEvent = 0;
  for (size_t i = 0; i < numSlice; i++)
  {
    EventSlice es(EM_Slice_Thickness_);
    es.it_begin_ = it_tmp;
    ros::Time t_end((*it_tmp)->ts.toSec() + es.SLICE_THICKNESS_);
    es.it_end_ = tools::EventVecPtr_lower_bound(vEventsPtr_left_, t_end);
    if (es.it_end_ == vEventsPtr_left_.end())
      es.it_end_--;
    es.numEvents_ = std::distance(es.it_begin_, es.it_end_) + 1;
    auto it_median = es.it_begin_;
    std::advance(it_median, es.numEvents_ / 2);
    es.t_median_ = (*it_median)->ts;
    getPoseAt(es.t_median_, es.transf_, dvs_frame_id_);
    eventSlices.push_back(es);
    it_tmp = es.it_end_;
    totalNumEvent += es.numEvents_;
    it_tmp++;
    if (it_tmp == vEventsPtr_left_.end())
      break;
    //    LOG(INFO) << "num of events in this slice[" << i << "]: " << eventSlices[i].numEvents_;
  }
  //  LOG(INFO) << "eventSlices: " << eventSlices.size();
}

void esvo_MVSMono::createEdgeMask(
  std::vector<dvs_msgs::Event *> &vEventsPtr,
  PerspectiveCamera::Ptr &camPtr,
  cv::Mat& edgeMap,
  std::vector<std::pair<size_t, size_t> >& vEdgeletCoordinates,
  bool bUndistortEvents,
  size_t radius)
{
  size_t col = camPtr->width_;
  size_t row = camPtr->height_;
  int dilate_radius = (int) radius;
  edgeMap = cv::Mat(cv::Size(col, row), CV_8UC1, cv::Scalar(0));
  vEdgeletCoordinates.reserve(col*row);

  auto it_tmp = vEventsPtr.begin();
  while (it_tmp != vEventsPtr.end())
  {
    // undistortion + rectification
    Eigen::Matrix<double,2,1> coor;
    if(bUndistortEvents)
      coor = camPtr->getRectifiedUndistortedCoordinate((*it_tmp)->x, (*it_tmp)->y);
    else
      coor = Eigen::Matrix<double,2,1>((*it_tmp)->x, (*it_tmp)->y);

    // assign
    int xcoor = std::floor(coor(0));
    int ycoor = std::floor(coor(1));

    for(int dy = -dilate_radius; dy <= dilate_radius; dy++)
      for(int dx = -dilate_radius; dx <= dilate_radius; dx++)
      {
        int x = xcoor + dx;
        int y = ycoor + dy;
        if(x < 0 || x >= col || y < 0 || y >= row)
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
  std::vector<dvs_msgs::Event *>& vAllEventsPtr,
  cv::Mat& mask,
  size_t row, size_t col)
{
  cv::Mat eventMap;
  visualizor_.plot_eventMap(vAllEventsPtr, eventMap, row, col);
  cv::medianBlur(eventMap, mask, 3);
}

void esvo_MVSMono::extractDenoisedEvents(
  std::vector<dvs_msgs::Event *> &vCloseEventsPtr,
  std::vector<dvs_msgs::Event *> &vEdgeEventsPtr,
  cv::Mat& mask,
  size_t maxNum)
{
  vEdgeEventsPtr.reserve(vCloseEventsPtr.size());
  for(size_t i = 0;i < vCloseEventsPtr.size();i++)
  {
    if(vEdgeEventsPtr.size() >= maxNum)
      break;
    size_t x = vCloseEventsPtr[i]->x;
    size_t y = vCloseEventsPtr[i]->y;
    if(mask.at<uchar>(y,x) == 255)
      vEdgeEventsPtr.push_back(vCloseEventsPtr[i]);
  }
}

}// esvo_core