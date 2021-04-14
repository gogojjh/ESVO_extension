#include <esvo_core/container/SensorSystem.h>
#include <opencv2/core/eigen.hpp>
#include <glog/logging.h>

namespace esvo_core
{
  namespace container
  {
    SensorSystem::SensorSystem(const std::string &calibInfoDir, bool bPrintCalibInfo)
    {
      loadCalibInfo(calibInfoDir, bPrintCalibInfo);
    }
    
    SensorSystem::~SensorSystem() {}

    void SensorSystem::loadCalibInfo(const std::string &sensorSystemDir, bool bPrintCalibInfo)
    {
      const std::string ext_calib_dir(sensorSystemDir + "/extrinsics.yaml");
      YAML::Node extCalibInfo = YAML::LoadFile(ext_calib_dir);

      // load calib (left)
      std::vector<double> vT_left_davis_lidar;
      std::vector<double> vT_gt_left_davis;

      vT_left_davis_lidar = extCalibInfo["T_left_davis_lidar"]["data"].as<std::vector<double>>();
      T_left_davis_lidar_ = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>(vT_left_davis_lidar.data());

      vT_gt_left_davis = extCalibInfo["T_gt_left_davis"]["data"].as<std::vector<double>>();
      T_gt_left_davis_ = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>(vT_gt_left_davis.data());

      if (bPrintCalibInfo)
        printCalibInfo();
    }

    void SensorSystem::printCalibInfo()
    {
      LOG(INFO) << "============================================" << std::endl;
      LOG(INFO) << "Extrinsic" << std::endl;
      LOG(INFO) << "--T_left_davis_lidar:\n" << T_left_davis_lidar_;
      LOG(INFO) << "--T_gt_left_davis:\n" << T_gt_left_davis_;
      LOG(INFO) << "============================================" << std::endl;
    }

  } // namespace container

} // namespace esvo_core
