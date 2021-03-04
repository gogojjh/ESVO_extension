#ifndef INITIAL_MOTION_ESTIMATOR_H_
#define INITIAL_MOTION_ESTIMATOR_H_

#include <glog/logging.h>

#include <vector>

#include <ros/ros.h>
#include <dvs_msgs/Event.h>

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_blas.h>

#define INI_MOT_EST_DEBUG

const int INI_NUM_EVENTS_INVOLVE = 20000;
const double INI_DOWNSAMPLE_RATE = 1.0;
const int INI_MAX_ITERATION = 10;
const double INI_DEPTH = 1.0;
const double INI_TIME_INTERVAL = 0.001;

class CMSummary
{
public:
    size_t iteration;
    double initial_cost, final_cost;
    int status;
};

enum
{
    BEZIER_LINER, // 0
    BEZIER_QUADRATIC,
    HOMO_TRANSFORMATION
}; // motion model

enum
{
    VARIANCE_CONTRAST, // 0
    MEAN_SQUARE_CONTRAST
}; // contrast definition

/**
 * @brief Options and Auxiliary data structure for optimization algorithm
 */
class MCAuxdata
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MCAuxdata(double *ref_time_, std::vector<Eigen::Vector4d> *events_coor_,
              cv::Size img_size_, Eigen::Matrix3d K_,
              int contrast_measure_, bool use_polarity_, double blur_sigma_)
        : ref_time(ref_time_),
          events_coor(events_coor_),
          img_size(img_size_),
          K(K_),
          contrast_measure(contrast_measure_),
          use_polarity(use_polarity_),
          blur_sigma(blur_sigma_)
    {}

    MCAuxdata()
        : ref_time(nullptr),
          events_coor(nullptr),
          img_size(cv::Size(0, 0)),
          K(Eigen::Matrix3d::Zero()),
          contrast_measure(VARIANCE_CONTRAST),
          use_polarity(false),
          blur_sigma(1.0)
    {}

    double *ref_time;
    std::vector<Eigen::Vector4d> *events_coor;

    cv::Size img_size;
    Eigen::Matrix3d K;
    
    int contrast_measure;
    bool use_polarity;
    double blur_sigma;
};

/** contrast maximization **/
double contrast_MeanSquare(const cv::Mat &image);
double contrast_Variance(const cv::Mat &image);
double computeContrast(const cv::Mat &image, const int contrast_measure);
void computeImageOfWarpedEvents(const double *v, MCAuxdata *poAux_data, cv::Mat *image_warped, double &energy);

/** gsl function **/
double contrast_f_numerical(const gsl_vector *v, void *adata);
void contrast_fdf_numerical(const gsl_vector *v, void *adata, double *f, gsl_vector *df);
void contrast_df_numerical(const gsl_vector *v, void *adata, gsl_vector *df);
double vs_gsl_Gradient_ForwardDiff(
    const gsl_vector *x,                               /**< [in] Point at which the gradient is to be evaluated */
    void *data,                                        /**< [in] Optional parameters passed directly to the function func_f */
    double (*func_f)(const gsl_vector *x, void *data), /**< [in] User-supplied routine that returns the value of the function at x */
    gsl_vector *J,                                     /**< [out] Gradient vector (same length as x) */
    double dh                                          /**< [in] Increment in variable for numerical differentiation */
);
double vs_gsl_Gradient_Analytic(
    const gsl_vector *x,
    void *data,
    double (*func_f)(const gsl_vector *x, void *data),
    gsl_vector *J,
    double dh);

class InitialMotionEstimator
{
public:
    InitialMotionEstimator();
    ~InitialMotionEstimator();

    bool setProblem(const double &curTime,
                    std::vector<Eigen::Vector4d> &vALLEvents,
                    const size_t &width,
                    const size_t &height,
                    const Eigen::Matrix3d &K);

    /** problem solver **/
    CMSummary solve();

    /** result **/
    Eigen::Matrix4d getMotion(const double &startTime, const double &endTime);
    void concatHorizontal(const cv::Mat &A, const cv::Mat &B, cv::Mat *C);   
    cv::Mat drawMCImage();

    /** online data **/
    double *x_, *x_last_; // tx, ty, rz
    double curTime_, prevTime_;
    // Eigen::MatrixXd TS_metric_;
    // std::vector<Eigen::Vector4d> vEdgeletCoordinates_;
    MCAuxdata MCAuxdata_;
};



#endif



