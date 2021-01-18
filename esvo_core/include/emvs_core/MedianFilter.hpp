#ifndef MEDIAN_FILTER_HPP_
#define MEDIAN_FILTER_HPP_

#include <opencv2/core/core.hpp>


void huangMedianFilter(const cv::Mat& img, cv::Mat& out_img, const cv::Mat& mask, int patch_size);

#endif