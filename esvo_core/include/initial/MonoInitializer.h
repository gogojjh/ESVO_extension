#ifndef MONO_INITIALIZER_H_
#define MONO_INITIALIZER_H_

#include <glog/logging.h>

#include <vector>
#include <cstdio>
#include <iostream>
#include <queue>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

// #include <initial/ModelSelector.h>
#include <initial/feature_manager.h>
// #include <initial/initial_sfm.h>
// #include <initial/solve_5pts.h>

template <typename DataType>
void reduceVector(std::vector<DataType> &v, std::vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

class MonoInitializer
{
public:
    MonoInitializer();
    ~MonoInitializer() = default;

    void setMask();
    bool inBorder(const cv::Point2f &pt);
    void trackImage(const double &t, const cv::Mat &img, const Eigen::Matrix3d &K);

    void addFeatureCheckParallax(const size_t &frame_count);
    bool initialize(const Eigen::Matrix3d &K);
    void relativePose(Eigen::Matrix3d &relative_R, Eigen::Vector3d &relative_T, int &l);

    void drawTrack();
    inline cv::Mat getTrackImage() const
    {
        return img_track_vis_;
    }
    
private:
    int row_, col_;
    cv::Mat mask_;
    cv::Mat prev_img_, cur_img_;
    cv::Mat img_track_vis_;
    std::vector<cv::Point2f> n_pts_;
    std::vector<cv::Point2f> prev_pts_, cur_pts_;
    std::vector<cv::Point2f> prev_un_pts_, cur_un_pts_;
    std::vector<int> ids_;
    std::vector<int> track_cnt_;

    std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> feature_vec_;
    double prev_time_, cur_time_;
    bool has_prediction_;
    int n_id_;

    int focal_length_;
    int frame_count_;
    // FeatureManager f_manager_;
    // MotionEstimator m_estimator_;

    // ModelSelector modelSelector_;

    bool isLargeParallax_;
};

#endif