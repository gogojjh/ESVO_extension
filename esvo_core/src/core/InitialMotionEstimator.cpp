#include "esvo_core/core/InitialMotionEstimator.h"

// ****************************************************************
// GSL-related functions
// ****************************************************************
/**
 * @brief Define loss function: Mean Square, Variance
 */
double contrast_MeanSquare(const cv::Mat &image)
{
    double contrast = cv::norm(image, cv::NORM_L2SQR) / static_cast<double>(image.rows * image.cols);
    return contrast;
}

double contrast_Variance(const cv::Mat &image)
{
    // Compute variance of the image
    cv::Scalar mean, stddev;
    cv::meanStdDev(image, mean, stddev);
    double contrast = stddev.val[0] * stddev.val[0];
    return contrast;
}

double computeContrast(const cv::Mat &image, const int contrast_measure)
{
    // Branch according to contrast measure
    double contrast;
    switch (contrast_measure)
    {
    case MEAN_SQUARE_CONTRAST:
        contrast = contrast_MeanSquare(image);
        break;
    default:
        contrast = contrast_Variance(image);
        break;
    }
    return contrast;
}

void computeImageOfWarpedEvents(const double *v,
                                MCAuxdata *poAux_data,
                                cv::Mat *image_warped)
{
    CHECK_GT(*(poAux_data->ref_time), poAux_data->events_coor->back()[2]);

    const int width = poAux_data->img_size.width;
    const int height = poAux_data->img_size.height;
    *image_warped = cv::Mat::zeros(poAux_data->img_size, CV_64FC1);

    const Eigen::Matrix3d K = poAux_data->K;
    std::vector<std::pair<double, Eigen::Matrix3d>> st_Hinv;
    double t_begin = poAux_data->events_coor->front()[2];
    double t_end = *(poAux_data->ref_time);
    double t_cur = t_begin;
    // the last as the reference time
    while (t_cur < t_end + INI_TIME_INTERVAL)
    {
        double dt = t_end - t_cur;
        double tx = -1 / INI_DEPTH * v[0] * dt;
        double ty = -1 / INI_DEPTH * v[1] * dt;
        double theta = v[2] * dt;
        Eigen::Matrix3d H_inv;
        H_inv << cos(theta), -sin(theta), tx,
            sin(theta), cos(theta), ty,
            0, 0, 1;
        H_inv = K * H_inv * K.inverse();
        st_Hinv.emplace_back(t_cur, H_inv);
        t_cur += INI_TIME_INTERVAL;
    }
    // LOG(INFO) << "H_inv size: " << st_Hinv.size(); // 60

    // size_t i = 0;
    // std::vector<size_t> mEventIdx;
    // mEventIdx.reserve(poAux_data->events_coor->size());
    // for (const dvs_msgs::Event &ev : *(poAux_data->events_coor))
    // {
    //     while ((ev.ts.toSec() > st_Hinv[i].first) && (i < st_Hinv.size() - 1))
    //         i++;
    //     mEventIdx.push_back(i);
    // }

    size_t j = 0;
    for (size_t i = 0; i < poAux_data->events_coor->size(); i++)
    {
        const Eigen::Vector4d &ev = (*poAux_data->events_coor)[i];
        while ((ev[2] > st_Hinv[j].first) && (j < st_Hinv.size() - 1))
            j++;

        Eigen::Vector3d p(ev[0], ev[1], 1);
        Eigen::Vector3d warp_p = st_Hinv[j].second * p;
        warp_p /= warp_p(2);

        const float polarity = (poAux_data->use_polarity) ? 2.f * static_cast<float>(ev[3]) - 1.f : 1.f;
        const int xx = warp_p.x(), yy = warp_p.y();
        if (1 <= xx && xx < width - 2 && 1 <= yy && yy < height - 2)
        {
            // Accumulate warped events on the IWE
            double dx = warp_p.x() - xx;
            double dy = warp_p.y() - yy;
            image_warped->at<double>(yy, xx) += polarity * (1 - dx) * (1 - dy);
            image_warped->at<double>(yy + 1, xx) += polarity * (1 - dx) * dy;
            image_warped->at<double>(yy, xx + 1) += polarity * dx * (1 - dy);
            image_warped->at<double>(yy + 1, xx + 1) += polarity * dx * dy;
        }
    }

    if (poAux_data->blur_sigma > 0)
    {
        cv::GaussianBlur(*image_warped, *image_warped, cv::Size(3, 3),
                         poAux_data->blur_sigma, poAux_data->blur_sigma);
    }
}

/**
 * @brief Main function used by optimization algorithm.
 * Maximize contrast, or equivalently, minimize (-contrast)
 */
double contrast_f_numerical(const gsl_vector *v, void *adata)
{
    // Extract auxiliary data of cost function
    MCAuxdata *poAux_data = (MCAuxdata *)adata;
    double x[3] = {gsl_vector_get(v, 0), gsl_vector_get(v, 1), gsl_vector_get(v, 2)};
    cv::Mat image_warped;
    computeImageOfWarpedEvents(x, poAux_data, &image_warped);
    double contrast = computeContrast(image_warped, poAux_data->contrast_measure);
    return -contrast;
}

void contrast_fdf_numerical(const gsl_vector *v, void *adata, double *f, gsl_vector *df)
{
    // Finite difference approximation
    *f = vs_gsl_Gradient_ForwardDiff(v, adata, contrast_f_numerical, df, 1e-1);

    // FILL IN
    // use the derived Jacobian to replace the forward_diff Jacobian
    // *f = vs_gsl_Gradient_Analytic(v, adata, contrast_f_numerical, df, 1e0);
}

void contrast_df_numerical(const gsl_vector *v, void *adata, gsl_vector *df)
{
    double cost;
    contrast_fdf_numerical(v, adata, &cost, df);
}

double vs_gsl_Gradient_ForwardDiff(
    const gsl_vector *x,                               /**< [in] Point at which the gradient is to be evaluated */
    void *data,                                        /**< [in] Optional parameters passed directly to the function func_f */
    double (*func_f)(const gsl_vector *x, void *data), /**< [in] User-supplied routine that returns the value of the function at x */
    gsl_vector *J,                                     /**< [out] Gradient vector (same length as x) */
    double dh = 1e-6                                   /**< [in] Increment in variable for numerical differentiation */
)
{
    // Evaluate vector function at x
    double fx = func_f(x, data);

    // Clone the parameter vector x
    gsl_vector *xh = gsl_vector_alloc(J->size);
    gsl_vector_memcpy(xh, x);

    for (int j = 0; j < J->size; j++)
    {
        gsl_vector_set(xh, j, gsl_vector_get(x, j) + dh); // Take a (forward) step in the current dimension
        double fh = func_f(xh, data);                     // Evaluate vector function at new x
        gsl_vector_set(J, j, fh - fx);                    // Finite difference approximation (except for 1/dh factor)
        gsl_vector_set(xh, j, gsl_vector_get(x, j));      // restore original value of the current variable
    }
    gsl_vector_scale(J, 1.0 / dh);
    gsl_vector_free(xh);
    return fx;
}

// TODO: provide the expression of Jacobians
double vs_gsl_Gradient_Analytic(
    const gsl_vector *x,                               /**< [in] Point at which the gradient is to be evaluated */
    void *data,                                        /**< [in] Optional parameters passed directly to the function func_f */
    double (*func_f)(const gsl_vector *x, void *data), /**< [in] User-supplied routine that returns the value of the function at x */
    gsl_vector *J,                                     /**< [out] Gradient vector (same length as x) */
    double dh = 1e-6                                   /**< [in] Increment in variable for numerical differentiation */
)
{
    // Evaluate vector function at x
    double fx = func_f(x, data);

    // Clone the parameter vector x
    gsl_vector *xh = gsl_vector_alloc(J->size);
    gsl_vector_memcpy(xh, x);

    for (int j = 0; j < J->size; j++)
    {
        // FILL IN
        // set the values of J
        // gsl_vector_set(J, j,
    }
    gsl_vector_free(xh);
    return fx;
}

// ****************************************************************
// InitialMotionEstimator
// ****************************************************************
InitialMotionEstimator::InitialMotionEstimator()
{
    x_ = new double[3];
    x_last_ = new double[3];
    x_[0] = x_[1] = x_[2] = 0.0;
    x_last_[0] = x_last_[1] = x_last_[2] = 0.0;
    
    prevTime_ = curTime_ = 0.0;

    MCAuxdata_.contrast_measure = VARIANCE_CONTRAST;
    MCAuxdata_.use_polarity = false;
    MCAuxdata_.blur_sigma = 1.0;
}

InitialMotionEstimator::~InitialMotionEstimator()
{
    delete[] x_;
    delete[] x_last_;
}

bool InitialMotionEstimator::setProblem(const double &curTime,
                                          const Eigen::MatrixXd &TS,
                                          const std::vector<dvs_msgs::Event *> vALLEventsPtr,
                                          const esvo_core::container::PerspectiveCamera::Ptr &camPtr,
                                          bool bUndistortEvents)
{
    size_t step = size_t(1.0 / INI_DOWNSAMPLE_RATE);
    size_t col = camPtr->width_;
    size_t row = camPtr->height_;
    for (size_t i = 0; i < vALLEventsPtr.size(); i += step)
    {
        // undistortion + rectification
        Eigen::Matrix<double, 2, 1> coor;
        if (bUndistortEvents)
            coor = camPtr->getRectifiedUndistortedCoordinate(vALLEventsPtr[i]->x, vALLEventsPtr[i]->y);
        else
            coor = Eigen::Matrix<double, 2, 1>(vALLEventsPtr[i]->x, vALLEventsPtr[i]->y);
        Eigen::Vector4d tmp_coor;
        tmp_coor[0] = coor(0);
        tmp_coor[1] = coor(1);
        tmp_coor[2] = vALLEventsPtr[i]->ts.toSec();
        tmp_coor[3] = double(vALLEventsPtr[i]->polarity);
        vEdgeletCoordinates_.push_back(tmp_coor);
    }

    if (vEdgeletCoordinates_.size() / INI_DOWNSAMPLE_RATE < INI_NUM_EVENTS_INVOLVE)
    {
        // LOG(INFO) << "Not enough events for motion initialization: "
        //           << vEdgeletCoordinates_.size() << " < " << INI_NUM_EVENTS_INVOLVE;
        return false;
    }

    x_last_[0] = x_[0];
    x_last_[1] = x_[1];
    x_last_[2] = x_[2];
    prevTime_ = curTime_;

    curTime_ = curTime;
    TS_ = TS;

    MCAuxdata_.ref_time = &curTime_;
    MCAuxdata_.events_coor = &vEdgeletCoordinates_;
    MCAuxdata_.TS = &TS_;
    MCAuxdata_.img_size = cv::Size(row, col);
    MCAuxdata_.K = camPtr->K_;
    return true;
}

CMSummary InitialMotionEstimator::solve()
{
    const gsl_multimin_fdfminimizer_type *solver_type;
    solver_type = gsl_multimin_fdfminimizer_conjugate_fr; // Fletcher-Reeves conjugate gradient algorithm

    gsl_multimin_function_fdf solver_info;

    const int num_params = 3;  // Size of global flow
    solver_info.n = num_params;               // Size of the parameter vector
    solver_info.f = contrast_f_numerical;     // Cost function
    solver_info.df = contrast_df_numerical;   // Gradient of cost function
    solver_info.fdf = contrast_fdf_numerical; // Cost and gradient functions
    solver_info.params = &MCAuxdata_;         // Auxiliary data

    gsl_vector *vx = gsl_vector_alloc(num_params);
    gsl_vector_set(vx, 0, x_[0]); // x1
    gsl_vector_set(vx, 1, x_[1]); // y1
    gsl_vector_set(vx, 2, x_[2]); // x2

    gsl_multimin_fdfminimizer *solver = gsl_multimin_fdfminimizer_alloc(solver_type, num_params);
    const double initial_step_size = 10;
    double tol = 0.05;

    gsl_multimin_fdfminimizer_set(solver, &solver_info, vx, initial_step_size, tol);
    const double initial_cost = solver->f;

#ifdef INI_MOT_EST_DEBUG
    LOG(INFO) << "--- Initial cost = " << std::setprecision(8) << initial_cost
              << "; ini: " << x_[0] << "m/s " << x_[1] << "m/s " << x_[2] / M_PI * 180 << "deg/s";
#endif

    const int num_max_line_searches = INI_MAX_ITERATION;
    int status;
    const double epsabs_grad = 1e-3, tolfun = 1e-2;
    double cost_new = 1e9, cost_old = 1e9;
    size_t iter = 0;
    do
    {
        iter++;
        cost_old = cost_new;
        status = gsl_multimin_fdfminimizer_iterate(solver);
        //status == GLS_SUCCESS (0) means that the iteration reduced the function value

        if (status == GSL_SUCCESS)
        {
            //Test convergence due to stagnation in the value of the function
            cost_new = gsl_multimin_fdfminimizer_minimum(solver);
            if (fabs(1 - cost_new / (cost_old + 1e-7)) < tolfun)
            {
                // LOG(INFO) << "progress tolerance reached.";
                break;
            }
            else
                status = GSL_CONTINUE;
        }

        //Test convergence due to absolute norm of the gradient
        if (GSL_SUCCESS == gsl_multimin_test_gradient(solver->gradient, epsabs_grad))
        {
            // LOG(INFO) << "gradient tolerance reached.";
            break;
        }

        if (status != GSL_CONTINUE)
        {
            // The iteration was not successful (did not reduce the function value)
            // LOG(INFO) << "stopped iteration; status = " << status;
            break;
        }
    } while (status == GSL_CONTINUE && iter < num_max_line_searches);

    gsl_vector *final_x = gsl_multimin_fdfminimizer_x(solver);
    x_[0] = gsl_vector_get(final_x, 0);
    x_[1] = gsl_vector_get(final_x, 1);
    x_[2] = gsl_vector_get(final_x, 2);
    const double final_cost = gsl_multimin_fdfminimizer_minimum(solver);

#ifdef INI_MOT_EST_DEBUG
        LOG(INFO) << "--- Final cost   = " << std::setprecision(8) << final_cost 
                  << "; opt: " << x_[0] << "m/s " << x_[1] << "m/s " << x_[2] / M_PI * 180 << "deg/s";
#endif

    gsl_multimin_fdfminimizer_free(solver);
    gsl_vector_free(vx);

    CMSummary summary;
    summary.status = status;
    summary.iteration = iter;
    summary.initial_cost = initial_cost;
    summary.final_cost = final_cost;
    return summary;
}

Eigen::Matrix4d InitialMotionEstimator::getMotion()
{
    double dt;
    if (prevTime_ == 0)
        dt = curTime_ - vEdgeletCoordinates_.front()[2];
    else
        dt = curTime_ - prevTime_;
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    double tx = -1 / INI_DEPTH * x_[0] * dt;
    double ty = -1 / INI_DEPTH * x_[1] * dt;
    double theta = x_[2] * dt;
    Eigen::Matrix3d R;
    R << cos(theta), -sin(theta), 0,
         sin(theta), cos(theta), 0,
         0, 0, 1;
    Eigen::Vector3d t(tx, ty, 0);
    T.topLeftCorner<3, 3>() = R;
    T.topRightCorner<3, 1>() = t;
    return T;
}

/**
 * \brief Concatenate two matrices horizontally
 * \param[in] Mat A and B
 * \param[out] Mat C = [A, B]
*/
void InitialMotionEstimator::concatHorizontal(const cv::Mat &A, const cv::Mat &B, cv::Mat *C)
{
    CHECK_EQ(A.rows, B.rows) << "Input arguments must have same number of rows";
    CHECK_EQ(A.type(), B.type()) << "Input arguments must have the same type";
    cv::hconcat(A, B, *C);
}

cv::Mat InitialMotionEstimator::drawMCImage()
{
    cv::Mat image_original, image_warped, image_stacked;
    double x_ini[4] = {0., 0., 0., 0.};
    computeImageOfWarpedEvents(x_ini, &MCAuxdata_, &image_original);
    computeImageOfWarpedEvents(x_, &MCAuxdata_, &image_warped);
    concatHorizontal(image_original, image_warped, &image_stacked);
    if (MCAuxdata_.use_polarity)
    {
        // Visualize the image of warped events with the zero always at the mean grayscale level
        const float bmax = 5.f;
        image_stacked = (255.0f / (2.0f * bmax)) * (image_stacked + bmax);
    }
    else
    {
        // Scale the image to full range [0,255]
        cv::normalize(image_stacked, image_stacked, 0.f, 255.0f, cv::NORM_MINMAX, CV_32FC1);
        image_stacked = 255.0f - image_stacked; // invert "color": dark events over white background
    }
    return image_stacked;
}