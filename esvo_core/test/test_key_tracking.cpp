
// ****************************************************************
// ***************** Nonliner optimization using Eigen
// https://eigen.tuxfamily.org/dox/unsupported/group__NonLinearOptimization__Module.html
// ****************************************************************
#include <unsupported/Eigen/NonLinearOptimization>
bool RegProblemSolverLM::solve_numerical()
{
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<RegProblemLM>, double> lm(*numDiff_regProblemPtr_.get());
    lm.resetParameters();
    lm.parameters.ftol = 1e-3;
    lm.parameters.xtol = 1e-3;
    lm.parameters.maxfev = rpConfigPtr_->MAX_ITERATION_ * 8;

    size_t iteration = 0;
    size_t nfev = 0;
    while (true)
    {
        if (iteration >= rpConfigPtr_->MAX_ITERATION_)
            break;
        numDiff_regProblemPtr_->setStochasticSampling(
            (iteration % numDiff_regProblemPtr_->numBatches_) * rpConfigPtr_->BATCH_SIZE_, rpConfigPtr_->BATCH_SIZE_);
        Eigen::VectorXd x(6);
        x.fill(0.0);
        if (lm.minimizeInit(x) == Eigen::LevenbergMarquardtSpace::ImproperInputParameters)
        {
            LOG(ERROR) << "ImproperInputParameters for LM (Tracking)." << std::endl;
            return false;
        }

        Eigen::LevenbergMarquardtSpace::Status status = lm.minimizeOneStep(x);
        numDiff_regProblemPtr_->addMotionUpdate(x);

        iteration++;
        nfev += lm.nfev;
        if (status == 2 || status == 3)
            break;
    }

    LOG(INFO) << "LM Finished ...................";
    numDiff_regProblemPtr_->setPose();
    lmStatics_.nPoints_ = numDiff_regProblemPtr_->numPoints_;
    lmStatics_.nfev_ = nfev;
    lmStatics_.nIter_ = iteration;
}

// ****************************************************************
// ***************** Tracking
// ****************************************************************
/**
 * @brief: Setup the optimization problem 
 * ref: 3D map
 * cur: Timesurface Map at the latest time
 */
void RegProblemLM::setProblem(RefFrame *ref, CurFrame *cur, bool bComputeGrad)
{
    ref_ = ref;
    cur_ = cur;
    T_world_ref_ = ref_->tr_.getTransformationMatrix();
    T_world_left_ = cur_->tr_.getTransformationMatrix();
    Eigen::Matrix4d T_ref_left = T_world_ref_.inverse() * T_world_left_; // the initial guess
    R_ = T_ref_left.block<3, 3>(0, 0);
    t_ = T_ref_left.block<3, 1>(0, 3);
    Eigen::Matrix3d R_world_ref = T_world_ref_.block<3, 3>(0, 0);
    Eigen::Vector3d t_world_ref = T_world_ref_.block<3, 1>(0, 3);

    // load ref's pointcloud tp vResItem
    ResItems_.clear();
    numPoints_ = ref_->vPointXYZPtr_.size();
    if (numPoints_ > rpConfigPtr_->MAX_REGISTRATION_POINTS_)
        numPoints_ = rpConfigPtr_->MAX_REGISTRATION_POINTS_;
    ResItems_.resize(numPoints_);

    for (size_t i = 0; i < numPoints_; i++)
    {
        bool bStochasticSampling = true;
        if (bStochasticSampling)
            std::swap(ref->vPointXYZPtr_[i], ref->vPointXYZPtr_[i + rand() % (ref->vPointXYZPtr_.size() - i)]);
        Eigen::Vector3d p_tmp((double)ref->vPointXYZPtr_[i]->x,
                              (double)ref->vPointXYZPtr_[i]->y,
                              (double)ref->vPointXYZPtr_[i]->z);
        Eigen::Vector3d p_cam = R_world_ref.transpose() * (p_tmp - t_world_ref);
        ResItems_[i].initialize(p_cam(0), p_cam(1), p_cam(2)); //, var);
    }
    // for stochastic sampling
    numBatches_ = std::max(ResItems_.size() / rpConfigPtr_->BATCH_SIZE_, (size_t)1);

    // load cur's info
    pTsObs_ = cur->pTsObs_;
    pTsObs_->getTimeSurfaceNegative(rpConfigPtr_->kernelSize_);

    // set fval dimension
    resetNumberValues(numPoints_ * patchSize_);
    if (bPrint_)
        LOG(INFO) << "RegProblemLM::setProblem succeeds.";
}

/**
 * @brief: Using multiple threads to compute the residuals
 */
void RegProblemLM::thread(Job &job) const
{
    // load info from job
    ResidualItems &vRI = *job.pvRI_;
    const TimeSurfaceObservation &TsObs = *job.pTsObs_;
    const Eigen::Matrix4d &T_left_ref = *job.T_left_ref_;
    size_t i_thread = job.i_thread_;
    size_t numPoint = vRI.size();
    size_t wx = rpConfigPtr_->patchSize_X_;
    size_t wy = rpConfigPtr_->patchSize_Y_;
    size_t residualDim = wx * wy;

    // calculate point-wise spatio-temporal residual
    // the residual can be either a scalr or a vector, up to the residualDim.
    for (size_t i = i_thread; i < numPoint; i += NUM_THREAD_)
    {
        ResidualItem &ri = vRI[i];
        ri.residual_ = Eigen::VectorXd(residualDim);
        Eigen::Vector2d x1_s;
        if (!reprojection(ri.p_, T_left_ref, x1_s))
            ri.residual_.setConstant(255.0);
        else
        {
            Eigen::MatrixXd tau1;
            if (patchInterpolation(TsObs.TS_negative_left_, x1_s, tau1))
            {
                for (size_t y = 0; y < wy; y++)
                    for (size_t x = 0; x < wx; x++)
                    {
                        size_t index = y * wx + x;
                        ri.residual_[index] = tau1(y, x);
                    }
            }
            else
                ri.residual_.setConstant(255.0);
        }
    }
}

/**
 * @brief: compute the residuals based on the negative Time Surface Map
 * equ. (17)
 */
bool RegProblemLM::patchInterpolation(const Eigen::MatrixXd &img,
                                      const Eigen::Vector2d &location, // positions of the reprojected map point
                                      Eigen::MatrixXd &patch, // residuals are represented by a patch
                                      bool debug) const
{
    int wx = rpConfigPtr_->patchSize_X_;
    int wy = rpConfigPtr_->patchSize_Y_;
    // compute SrcPatch_UpLeft coordinate and SrcPatch_DownRight coordinate
    // check patch bourndary is inside img boundary
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

    double q1 = upper_indices.second - double_indices[1];
    double q2 = double_indices[1] - lower_indices.second;
    double q3 = upper_indices.first - double_indices[0];
    double q4 = double_indices[0] - lower_indices.first;

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
 * @brief: Pose variables: dx = [tx, ty, tz, theta_x, theta_y, theta_z]
 */
void RegProblemLM::getWarpingTransformation(Eigen::Matrix4d &warpingTransf,
                                            const Eigen::Matrix<double, 6, 1> &x) const
{
    // To calcuate R_cur_ref, t_cur_ref
    Eigen::Matrix3d R_cur_ref;
    Eigen::Vector3d t_cur_ref;
    // get delta cayley paramters (this corresponds to the delta motion of the ref frame)
    Eigen::Vector3d dc = x.block<3, 1>(0, 0);
    Eigen::Vector3d dt = x.block<3, 1>(3, 0);
    // add rotation
    Eigen::Matrix3d dR = tools::cayley2rot(dc);
    Eigen::Matrix3d newR = R_.transpose() * dR.transpose();
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(newR, Eigen::ComputeFullU | Eigen::ComputeFullV);
    R_cur_ref = svd.matrixU() * svd.matrixV().transpose();
    if (R_cur_ref.determinant() < 0.0)
    {
        LOG(INFO) << "oops the matrix is left-handed\n";
        exit(-1);
    }
    t_cur_ref = -R_cur_ref * (dt + dR * t_);
    warpingTransf.block<3, 3>(0, 0) = R_cur_ref;
    warpingTransf.block<3, 1>(0, 3) = t_cur_ref;
}

/**
 * @brief: Pose updates
 */
void RegProblemLM::addMotionUpdate(const Eigen::Matrix<double, 6, 1> &dx)
{
    // To update R_, t_
    Eigen::Vector3d dc = dx.block<3, 1>(0, 0);
    Eigen::Vector3d dt = dx.block<3, 1>(3, 0);
    // add rotation
    Eigen::Matrix3d dR = tools::cayley2rot(dc);
    Eigen::Matrix3d newR = dR * R_;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(newR, Eigen::ComputeFullU | Eigen::ComputeFullV);
    R_ = svd.matrixU() * svd.matrixV().transpose();
    t_ = dt + dR * t_;
}

void RegProblemLM::setPose()
{
    T_world_left_.block<3, 3>(0, 0) = T_world_ref_.block<3, 3>(0, 0) * R_;
    T_world_left_.block<3, 1>(0, 3) = T_world_ref_.block<3, 3>(0, 0) * t_ + T_world_ref_.block<3, 1>(0, 3);
    cur_->tr_ = Transformation(T_world_left_);
}