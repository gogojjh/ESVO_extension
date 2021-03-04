#include <initial/InitialMotionEstimator.h>

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
                                cv::Mat *image_warped,
                                double &cost)
{
    CHECK_GT(*(poAux_data->ref_time), poAux_data->events_coor->front()[2]);

    const int width = poAux_data->img_size.width;
    const int height = poAux_data->img_size.height;
    *image_warped = cv::Mat::zeros(poAux_data->img_size, CV_64FC1);

    const Eigen::Matrix3d K = poAux_data->K;
    std::vector<std::pair<double, Eigen::Matrix3d>> st_Hinv;
    double t_begin = poAux_data->events_coor->back()[2];
    double t_end = *(poAux_data->ref_time);
    double t_tmp = t_end;
    // the last as the reference time
    while (t_tmp >= t_begin - INI_TIME_INTERVAL)
    {
        double dt = t_end - t_tmp;

        // planar homography estimation
        // H(t) = R(t) - 1/d t(t) n^{T}
        double wx = v[0] * dt;
        double wy = v[1] * dt;
        double wz = v[2] * dt;
        double tx = v[3] * dt;
        double ty = v[4] * dt;
        double tz = v[5] * dt;

        // Eigen::Vector3d w(wx, wy, wz);
        // double theta = w.norm();
        // Eigen::Vector3d a = w / theta;
        // a.normalize();
        // Eigen::Matrix3d R;
        // R = Eigen::AngleAxisd(theta, a);
        Eigen::Matrix3d R;
        // R = Eigen::AngleAxisd(wz, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(wy, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(wx, Eigen::Vector3d::UnitX());
        R = Eigen::AngleAxisd(wz, Eigen::Vector3d::UnitZ());
        Eigen::Vector3d t(tx, ty, tz);
        // Eigen::Vector3d t(tx, ty, 0.0);
        Eigen::Vector3d n(0.0, 0.0, -1.0);
        Eigen::Matrix3d H = R - 1 / INI_DEPTH * t * n.transpose();
        // std::cout << std::endl << R << std::endl << std::endl << t.transpose() << std::endl;
        Eigen::Matrix3d H_inv = H.inverse();
        st_Hinv.emplace_back(t_tmp, H_inv);
        t_tmp -= INI_TIME_INTERVAL;
    }
    // LOG(INFO) << "H_inv size: " << st_Hinv.size(); // 60

    size_t j = 0;
    for (size_t i = 0; i < poAux_data->events_coor->size(); i++)
    {
        const Eigen::Vector4d &ev = (*poAux_data->events_coor)[i];
        while ((ev[2] < st_Hinv[j].first) && (j < st_Hinv.size() - 1))
            j++;

        Eigen::Vector3d p(ev[0], ev[1], 1);
        Eigen::Vector3d warp_p = st_Hinv[j].second * p;
        warp_p /= warp_p(2);

        const float polarity = (poAux_data->use_polarity) ? 2.f * static_cast<float>(ev[3]) - 1.f : 1.f;
        const int xx = static_cast<int>(warp_p.x()), yy = static_cast<int>(warp_p.y());
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
    // double x[3] = {gsl_vector_get(v, 0), gsl_vector_get(v, 1), gsl_vector_get(v, 2)};
    double x[6] = {
        gsl_vector_get(v, 0),
        gsl_vector_get(v, 1),
        gsl_vector_get(v, 2),
        gsl_vector_get(v, 3),
        gsl_vector_get(v, 4),
        gsl_vector_get(v, 5)};
    cv::Mat image_warped;
    double cost = 0;
    computeImageOfWarpedEvents(x, poAux_data, &image_warped, cost);
    double contrast = computeContrast(image_warped, poAux_data->contrast_measure);
    return -contrast;
    // return -cost;
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
    x_ = new double[6];
    x_last_ = new double[6];
    x_[0] = x_[1] = x_[2] = x_[3] = x_[4] = x_[5] = 0.0;
    x_last_[0] = x_last_[1] = x_last_[2] = x_last_[3] = x_last_[4] = x_last_[5] = 0.0;
    prevTime_ = curTime_ = 0.0;
}

InitialMotionEstimator::~InitialMotionEstimator()
{
    delete[] x_;
    delete[] x_last_;
}

bool InitialMotionEstimator::setProblem(const double &curTime,
                                        std::vector<Eigen::Vector4d> &vALLEvents,
                                        const size_t &width,
                                        const size_t &height,
                                        const Eigen::Matrix3d &K)
{
    x_last_[0] = x_[0];
    x_last_[1] = x_[1];
    x_last_[2] = x_[2];
    x_last_[3] = x_[3];
    x_last_[4] = x_[4];
    x_last_[5] = x_[5];
    // prevTime_ = curTime_;

    curTime_ = curTime;
    MCAuxdata_ = MCAuxdata(&curTime_, &vALLEvents, cv::Size(width, height), K,
                           VARIANCE_CONTRAST, false, 1.0);
    return true;
}

CMSummary InitialMotionEstimator::solve()
{
    const gsl_multimin_fdfminimizer_type *solver_type;
    solver_type = gsl_multimin_fdfminimizer_conjugate_fr; // Fletcher-Reeves conjugate gradient algorithm

    gsl_multimin_function_fdf solver_info;

    const int num_params = 6;  // Size of global flow
    solver_info.n = num_params;               // Size of the parameter vector
    solver_info.f = contrast_f_numerical;     // Cost function
    solver_info.df = contrast_df_numerical;   // Gradient of cost function
    solver_info.fdf = contrast_fdf_numerical; // Cost and gradient functions
    solver_info.params = &MCAuxdata_;         // Auxiliary data

    gsl_vector *vx = gsl_vector_alloc(num_params);
    gsl_vector_set(vx, 0, x_[0]); 
    gsl_vector_set(vx, 1, x_[1]); 
    gsl_vector_set(vx, 2, x_[2]);
    gsl_vector_set(vx, 3, x_[3]); 
    gsl_vector_set(vx, 4, x_[4]);
    gsl_vector_set(vx, 5, x_[5]);

    gsl_multimin_fdfminimizer *solver = gsl_multimin_fdfminimizer_alloc(solver_type, num_params);
    const double initial_step_size = 10;
    double tol = 0.05;

    gsl_multimin_fdfminimizer_set(solver, &solver_info, vx, initial_step_size, tol);
    const double initial_cost = solver->f;

#ifdef INI_MOT_EST_DEBUG
    std::cout << "--- Initial cost = " << std::setprecision(8) << initial_cost
              << "; w (deg/s): " << x_[0] / M_PI * 180 << ", " << x_[1] / M_PI * 180 << ", " << x_[2] / M_PI * 180
              << ", v (m/s): " << x_[3] << ", " << x_[4] << ", " << x_[5] << std::endl;
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

    const double final_cost = gsl_multimin_fdfminimizer_minimum(solver);
    if (final_cost < initial_cost)
    {
        gsl_vector *final_x = gsl_multimin_fdfminimizer_x(solver);
        x_[0] = gsl_vector_get(final_x, 0);
        x_[1] = gsl_vector_get(final_x, 1);
        x_[2] = gsl_vector_get(final_x, 2);
        x_[3] = gsl_vector_get(final_x, 3);
        x_[4] = gsl_vector_get(final_x, 4);
        x_[5] = gsl_vector_get(final_x, 5);
#ifdef INI_MOT_EST_DEBUG
        std::cout << "--- Final cost = " << std::setprecision(8) << final_cost
                  << "; w (deg/s):" << x_[0] / M_PI * 180 << ", " << x_[1] / M_PI * 180 << ", " << x_[2] / M_PI * 180
                  << ", v (m/s):" << x_[3] << ", " << x_[4] << ", " << x_[5] << std::endl;
#endif
    }
    else
    {
        LOG(INFO) << "--- Fail to optimize";
    }

    gsl_multimin_fdfminimizer_free(solver);
    gsl_vector_free(vx);

    CMSummary summary;
    summary.status = status;
    summary.iteration = iter;
    summary.initial_cost = initial_cost;
    summary.final_cost = final_cost;
    return summary;
}

Eigen::Matrix4d InitialMotionEstimator::getMotion(const double &startTime, const double &endTime)
{
    double dt = endTime - startTime;
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();

    double wx = x_[0] * dt;
    double wy = x_[1] * dt;
    double wz = x_[2] * dt;
    double tx = x_[3] * dt;
    double ty = x_[4] * dt;
    double tz = x_[5] * dt;

    // Eigen::Vector3d w(wx, wy, wz);
    // double theta = w.norm();
    // Eigen::Vector3d a = w / theta;
    // a.normalize();
    // Eigen::Matrix3d R; 
    // R = Eigen::AngleAxisd(theta, a);
    Eigen::Matrix3d R;
    // R = Eigen::AngleAxisd(wz, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(wy, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(wx, Eigen::Vector3d::UnitX());
    R = Eigen::AngleAxisd(wz, Eigen::Vector3d::UnitZ());
    Eigen::Vector3d t(tx, ty, tz);
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
    double x_ini[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double cost = 0;
    computeImageOfWarpedEvents(x_ini, &MCAuxdata_, &image_original, cost);
    computeImageOfWarpedEvents(x_, &MCAuxdata_, &image_warped, cost);
    if (MCAuxdata_.use_polarity)
    {
        // Visualize the image of warped events with the zero always at the mean grayscale level
        const float bmax = 5.f;
        image_original = (255.0f / (2.0f * bmax)) * (image_original + bmax);
        image_warped = (255.0f / (2.0f * bmax)) * (image_warped + bmax);
    }
    else
    {
        // Scale the image to full range [0,255]
        cv::normalize(image_original, image_original, 0.f, 255.0f, cv::NORM_MINMAX, CV_32FC1);
        image_original = 255.0f - image_original; // invert "color": dark events over white background
        cv::normalize(image_warped, image_warped, 0.f, 255.0f, cv::NORM_MINMAX, CV_32FC1);
        image_warped = 255.0f - image_warped; // invert "color": dark events over white background
    }
    concatHorizontal(image_original, image_warped, &image_stacked);
    return image_stacked;
}