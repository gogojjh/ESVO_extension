#ifndef MONO_INITIALIZER_H_
#define MONO_INITIALIZER_H_

#include <glog/logging.h>

#include <vector>
#include <cstdio>
#include <iostream>
#include <queue>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

class MonoInitializer
{
public:
    MonoInitializer();
    ~MonoInitializer() = default;

    void setMask();
    bool inBorder(const cv::Point2f &pt);
    void trackImage(const double &t, const cv::Mat &img, const Eigen::Matrix3d &K);

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

    std::map<int, std::pair<int, Eigen::Matrix<double, 5, 1> > > feature_vec_;

    double prev_time_, cur_time_;
    bool has_prediction_;
    int n_id_;
};

#endif