
// ****************************************************************
// ***************** Mapping
// ****************************************************************
/**
 * @brief: initialize the map using SGM
 */
bool esvo_Mapping::InitializationAtTime(const ros::Time &t)
{
    cv::Ptr<cv::StereoSGBM> sgbm_;

    // call SGM on the current Time Surface observation pair.
    cv::Mat dispMap, dispMap8;
    sgbm_->compute(TS_obs_.second.cvImagePtr_left_->image, TS_obs_.second.cvImagePtr_right_->image, dispMap);
    dispMap.convertTo(dispMap8, CV_8U, 255 / (num_disparities_ * 16.));

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
    // ...
}

/**
 * @brief: After initializing the map, several steps are done: 
 * 1. initialize local map 
 * 2. Optimizing depths using stereo observations (10ms) and Time Surface Map | equ.(3)
 * 3. Inverse Depth Fusion | equ. (13), Fig. 7
 * 4. Regularization 
 */
void esvo_Mapping::MappingAtTime(const ros::Time &t)
{
    /************************************************/
    /************ set the new DepthFrame ************/
    /************************************************/
    DepthFrame::Ptr depthFramePtr_new = std::make_shared<DepthFrame>(
        camSysPtr_->cam_left_ptr_->height_, camSysPtr_->cam_left_ptr_->width_);
    depthFramePtr_new->setId(TS_obs_.second.id_);
    depthFramePtr_new->setTransformation(TS_obs_.second.tr_);
    depthFramePtr_ = depthFramePtr_new;

    // ******************************** block matching
    std::vector<EventMatchPair> vEMP; // the container that stores the result of BM.
    ebm_.createMatchProblem(&TS_obs_, &st_map_, &vDenoisedEventsPtr_left_);
    ebm_.match_all_HyperThread(vEMP);

    /**************************************************************/
    /*************  Nonlinear Optimization & Fusion ***************/
    /**************************************************************/
    double t_optimization = 0;
    double t_solve, t_fusion, t_regularization;
    t_solve = t_fusion = t_regularization = 0;
    size_t numFusionCount = 0; // To count the total number of fusion (in terms of fusion between two estimates, i.e. a priori and a propagated one).
    tt_mapping.tic();

    // ******************************** Nonlinear opitmization
    std::vector<DepthPoint> vdp; // depth points on the current stereo observations
    vdp.reserve(vEMP.size());
    dpSolver_.solve(&vEMP, &TS_obs_, vdp); // hyper-thread version
    dpSolver_.pointCulling(vdp, stdVar_vis_threshold_, cost_vis_threshold_,
                           invDepth_min_range_, invDepth_max_range_);

    // ******************************** Fusion (strategy 1: const number of point)
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
        dqvDepthPoints_.push_back(vdp); // all depth points on historical stereo observations
        while (dqvDepthPoints_.size() > maxNumFusionFrames_)
            dqvDepthPoints_.pop_front();
    }
    else
    {
        LOG(INFO) << "Invalid FusionStrategy is assigned.";
        exit(-1);
    }

    // ******************************** apply fusion and count the total number of fusion.
    numFusionCount = 0;
    for (auto it = dqvDepthPoints_.rbegin(); it != dqvDepthPoints_.rend(); it++)
    {
        numFusionCount += dFusor_.update(*it, depthFramePtr_, fusion_radius_); // fusing new points to update the depthmap
    }
    depthFramePtr_->dMap_->clean(pow(stdVar_vis_threshold_, 2), age_vis_threshold_, invDepth_max_range_, invDepth_min_range_);

    // regularization
    if (bRegularization_)
    {
        tt_mapping.tic();
        dRegularizor_.apply(depthFramePtr_->dMap_);
        t_regularization = tt_mapping.toc();
    }
    // ...
}

/**
 * @brief: initialize depth by block matching along epipolar line
 */
bool esvo_core::core::EventBM::match_an_event(dvs_msgs::Event *pEvent,
                                              std::pair<size_t, size_t> &pDisparityBound,
                                              esvo_core::core::EventMatchPair &emPair)
{
    size_t lowDisparity = pDisparityBound.first;
    size_t upDisparity = pDisparityBound.second;

    // rectify and floor the coordinate
    Eigen::Vector2d x_rect = camSysPtr_->cam_left_ptr_->getRectifiedUndistortedCoordinate(pEvent->x, pEvent->y);
    if (!camSysPtr_->cam_left_ptr_->isValidPixel(x_rect))
        return false;

    // This is to avoid depth estimation happenning in the mask area.
    if (camSysPtr_->cam_left_ptr_->UndistortRectify_mask_(x_rect(1), x_rect(0)) <= 125)
        return false;

    Eigen::Vector2i x1(std::floor(x_rect(0)), std::floor(x_rect(1)));
    Eigen::Vector2i x1_left_top;
    if (!isValidPatch(x1, x1_left_top))
        return false;

    // extract the template patch in the left time_surface
    Eigen::MatrixXd patch_src = pStampedTsObs_->second.TS_left_.block(
        x1_left_top(1), x1_left_top(0), patch_size_Y_, patch_size_X_);

    if ((patch_src.array() < 1).count() > 0.95 * patch_src.size())
    {
        //    LOG(INFO) << "Low info-noise-ratio. @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@";
        infoNoiseRatioLowNum_++;
        return false;
    }
    // LOG(INFO) << "patch_src is extracted";

    // searching along the epipolar line (heading to the left direction)
    double min_cost = ZNCC_MAX_;
    Eigen::Vector2i bestMatch;
    size_t bestDisp;
    Eigen::MatrixXd patch_dst = Eigen::MatrixXd::Zero(patch_size_Y_, patch_size_X_);
    // coarse searching
    if (!epipolarSearching(min_cost, bestMatch, bestDisp, patch_dst,
                           lowDisparity, upDisparity, step_,
                           x1, patch_src, bUpDownConfiguration_))
    {
        //    LOG(INFO) << "Coarse searching fails #################################";
        coarseSearchingFailNum_++;
        return false;
    }
    // fine searching
    size_t fine_searching_start_pos = bestDisp - (step_ - 1) >= 0 ? bestDisp - (step_ - 1) : 0;
    if (!epipolarSearching(min_cost, bestMatch, bestDisp, patch_dst,
                           fine_searching_start_pos, bestDisp + (step_ - 1), 1,
                           x1, patch_src, bUpDownConfiguration_))
    {
        // This indicates the local minima is not surrounded by two neighbors with larger cost,
        // This case happens when the best match locates over/outside the boundary of the Time Surface.
        fineSearchingFailNum_++;
        //    LOG(INFO) << "fine searching fails ...............";
        return false;
    }

    // transfer best match to emPair
    if (min_cost <= ZNCC_Threshold_)
    {
        emPair.x_left_raw_ = Eigen::Vector2d((double)pEvent->x, (double)pEvent->y);
        emPair.x_left_ = x_rect;
        emPair.x_right_ = Eigen::Vector2d((double)bestMatch(0), (double)bestMatch(1));
        emPair.t_ = pEvent->ts;
        double disparity;
        if (bUpDownConfiguration_)
            disparity = x1(1) - bestMatch(1);
        else
            disparity = x1(0) - bestMatch(0);
        double depth = camSysPtr_->baseline_ * camSysPtr_->cam_left_ptr_->P_(0, 0) / disparity;

        auto st_map_iter = tools::StampTransformationMap_lower_bound(*pSt_map_, emPair.t_);
        if (st_map_iter == pSt_map_->end())
            return false;
        emPair.trans_ = st_map_iter->second;
        emPair.invDepth_ = 1.0 / depth;
        emPair.cost_ = min_cost;
        emPair.disp_ = disparity;
        return true;
    }
    else
    {
        //    LOG(INFO) << "BM fails because: " << min_cost << " > " << ZNCC_Threshold_;
        return false;
    }
}

/**
 * @brief: compute the residuals temporal residuals, optimize the inverse depth
 */
bool DepthProblem::patchInterpolation(const Eigen::MatrixXd &img,
                                      const Eigen::Vector2d &location,
                                      Eigen::MatrixXd &patch,
                                      bool debug) const
{
    int wx = dpConfigPtr_->patchSize_X_;
    int wy = dpConfigPtr_->patchSize_Y_;
    // compute SrcPatch_UpLeft coordinate and SrcPatch_DownRight coordinate
    // check patch boundary is inside img boundary
    Eigen::Vector2i SrcPatch_UpLeft, SrcPatch_DownRight;
    SrcPatch_UpLeft << floor(location[0]) - (wx - 1) / 2, floor(location[1]) - (wy - 1) / 2;
    SrcPatch_DownRight << floor(location[0]) + (wx - 1) / 2, floor(location[1]) + (wy - 1) / 2;

    if (SrcPatch_UpLeft[0] < 0 || SrcPatch_UpLeft[1] < 0)
    {
        if (debug)
        {
            LOG(INFO) << "patchInterpolation 1: " << SrcPatch_UpLeft.transpose();
        }
        return false;
    }
    if (SrcPatch_DownRight[0] >= img.cols() || SrcPatch_DownRight[1] >= img.rows())
    {
        if (debug)
        {
            LOG(INFO) << "patchInterpolation 2: " << SrcPatch_DownRight.transpose();
        }
        return false;
    }

    // compute q1 q2 q3 q4
    Eigen::Vector2d double_indices;
    double_indices << location[1], location[0];

    std::pair<int, int> lower_indices(floor(double_indices[0]), floor(double_indices[1]));
    std::pair<int, int> upper_indices(lower_indices.first + 1, lower_indices.second + 1);

    double q1 = upper_indices.second - double_indices[1]; // x
    double q2 = double_indices[1] - lower_indices.second; // x
    double q3 = upper_indices.first - double_indices[0];  // y
    double q4 = double_indices[0] - lower_indices.first;  // y

    // extract Src patch, size (wy+1) * (wx+1)
    int wx2 = wx + 1;
    int wy2 = wy + 1;
    if (SrcPatch_UpLeft[1] + wy >= img.rows() || SrcPatch_UpLeft[0] + wx >= img.cols())
    {
        if (debug)
        {
            LOG(INFO) << "patchInterpolation 3: " << SrcPatch_UpLeft.transpose()
                      << ", location: " << location.transpose()
                      << ", floor(location[0]): " << floor(location[0])
                      << ", (wx - 1) / 2: " << (wx - 1) / 2
                      << ", ans: " << floor(location[0]) - (wx - 1) / 2
                      << ", wx: " << wx << " wy: " << wy
                      << ", img.row: " << img.rows() << " img.col: " << img.cols();
        }
        return false;
    }
    Eigen::MatrixXd SrcPatch = img.block(SrcPatch_UpLeft[1], SrcPatch_UpLeft[0], wy2, wx2);

    // Compute R, size (wy+1) * wx.
    Eigen::MatrixXd R;
    R = q1 * SrcPatch.block(0, 0, wy2, wx) + q2 * SrcPatch.block(0, 1, wy2, wx);

    // Compute F, size wy * wx.
    patch = q3 * R.block(0, 0, wy, wx) + q4 * R.block(1, 0, wy, wx);
    return true;
}

/**
 * @brief: Each map point is modeled as a Student's t distrubtion, 
 * this function fused multiple map points if they stay near
 */
int DepthFusion::fusion(DepthPoint &dp_prop,
                        DepthMap::Ptr &dm,
                        int fusion_radius)
{
    int numFusion = 0;

    // get neighbour pixels involved in fusion
    std::vector<std::pair<size_t, size_t>> vpCoordinate; // pair: <row, col>
    if (fusion_radius == 0)
    {
        const size_t patchSize = 4;
        vpCoordinate.reserve(patchSize);
        size_t row_topleft = dp_prop.row();
        size_t col_topleft = dp_prop.col();
        for (int dy = 0; dy <= 1; dy++)
            for (int dx = 0; dx <= 1; dx++)
                vpCoordinate.push_back(std::make_pair(row_topleft + dy, col_topleft + dx));
    }
    else
    {
        const size_t patchSize = (2 * fusion_radius + 1) * (2 * fusion_radius + 1);
        vpCoordinate.reserve(patchSize);
        size_t row_centre = dp_prop.row();
        size_t col_centre = dp_prop.col();
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++)
                vpCoordinate.push_back(std::make_pair(row_centre + dy, col_centre + dx));
    }

    // fusion
    for (size_t i = 0; i < vpCoordinate.size(); i++)
    {
        size_t row = vpCoordinate[i].first;
        size_t col = vpCoordinate[i].second;
        if (!boundaryCheck(col, row, camSysPtr_->cam_left_ptr_->width_, camSysPtr_->cam_left_ptr_->height_))
            continue;

        // case 1: non-occupied
        if (!dm->exists(row, col))
        {
            DepthPoint dp_new(row, col);
            if (strcmp(dpConfigPtr_->LSnorm_.c_str(), "l2") == 0)
                dp_new.update(dp_prop.invDepth(), dp_prop.variance());
            else if (strcmp(dpConfigPtr_->LSnorm_.c_str(), "Tdist") == 0)
            {
                dp_new.update_studentT(dp_prop.invDepth(), dp_prop.scaleSquared(), dp_prop.variance(), dp_prop.nu());
            }
            else
                exit(-1);

            dp_new.residual() = dp_prop.residual();
            Eigen::Vector3d p_cam;
            camSysPtr_->cam_left_ptr_->cam2World(dp_new.x(), dp_prop.invDepth(), p_cam);
            dp_new.update_p_cam(p_cam);

            dm->set(row, col, dp_new);
        }
        else // case 2: occupied
        {
            bool bCompatibility = false;
            if (strcmp(dpConfigPtr_->LSnorm_.c_str(), "l2") == 0)
                bCompatibility = chiSquareTest(dp_prop.invDepth(), dm->at(row, col).invDepth(),
                                               dp_prop.variance(), dm->at(row, col).variance());
            else if (strcmp(dpConfigPtr_->LSnorm_.c_str(), "Tdist") == 0)
            {
                bCompatibility = studentTCompatibleTest(
                    dp_prop.invDepth(), dm->at(row, col).invDepth(), dp_prop.variance(), dm->at(row, col).variance());
            }
            else
                exit(-1);

            // case 2.1 compatible
            if (bCompatibility)
            {
                if (strcmp(dpConfigPtr_->LSnorm_.c_str(), "l2") == 0)
                    dm->get(row, col).update(dp_prop.invDepth(), dp_prop.variance());
                else if (strcmp(dpConfigPtr_->LSnorm_.c_str(), "Tdist") == 0)
                    dm->get(row, col).update_studentT(dp_prop.invDepth(), dp_prop.scaleSquared(), dp_prop.variance(), dp_prop.nu());
                else
                    exit(-1);

                dm->get(row, col).age()++;
                dm->get(row, col).residual() = min(dm->get(row, col).residual(), dp_prop.residual());
                Eigen::Vector3d p_update;
                camSysPtr_->cam_left_ptr_->cam2World(dm->get(row, col).x(), dp_prop.invDepth(), p_update);
                dm->get(row, col).update_p_cam(p_update);
                numFusion++;
            }
            else // case 2.2 not compatible
            {
                // consider occlusion (the pixel is already assigned with a point that is closer to the camera)
                if (dm->at(row, col).invDepth() - 2 * sqrt(dm->at(row, col).variance()) > dp_prop.invDepth())
                    continue;
                if (dp_prop.variance() < dm->at(row, col).variance() && dp_prop.residual() < dm->at(row, col).residual()) //&& other requirement? such as cost?
                {
                    dm->get(row, col) = dp_prop;
                }
            }
        }
    }
    return numFusion;
}

/**
 * @brief: regularization
 */
void DepthRegularization::apply(DepthMap::Ptr &depthMapPtr)