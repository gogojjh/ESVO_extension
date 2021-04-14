#ifndef ESVO_CORE_CONTAINER_SENSORSYSTEM_H
#define ESVO_CORE_CONTAINER_SENSORSYSTEM_H

#include <iostream>
#include <string>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>
#include <yaml-cpp/yaml.h>

namespace esvo_core
{
  namespace container
  {
    class SensorSystem
    {
    public:
      SensorSystem(const std::string &calibInfoDir, bool bPrintCalibInfo = false);
      virtual ~SensorSystem();
      using Ptr = std::shared_ptr<SensorSystem>;

      void loadCalibInfo(const std::string &sensorSystemDir, bool bPrintCalibInfo = false);
      void printCalibInfo();

      Eigen::Matrix<double, 4, 4> T_left_davis_lidar_, T_gt_left_davis_;
    };
  } // namespace container
} // namespace esvo_core

#endif //ESVO_CORE_CONTAINER_SENSORSYSTEM_H