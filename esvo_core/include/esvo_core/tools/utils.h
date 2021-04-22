#ifndef ESVO_CORE_TOOLS_UTILS_H
#define ESVO_CORE_TOOLS_UTILS_H

#include <Eigen/Eigen>
#include <iostream>

#include <cv_bridge/cv_bridge.h>

#include <tf/tf.h>
#include <tf/tfMessage.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>
#include <kindr/minimal/quat-transformation.h>

#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>

#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <esvo_core/container/SmartGrid.h>
#include <esvo_core/container/DepthPoint.h>
#include <esvo_core/tools/TicToc.h>

#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>

// esvo_core::Transformation tr;
// Eigen::Matrix4d T = tr.getTransformationMatrix();

using namespace std;
namespace esvo_core
{
  namespace tools
  {
// TUNE this according to your platform's computational capability.
#define NUM_THREAD_TRACKING 2
#define NUM_THREAD_MAPPING 4

    typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
    using RefPointCloudMap = std::map<ros::Time, PointCloud::Ptr>;

    typedef pcl::PointCloud<pcl::PointXYZI> PointCloudI;
    using RefPointCloudIMap = std::map<ros::Time, PointCloudI::Ptr>;
    using RefPointCloudIPair = std::pair<ros::Time, PointCloudI::Ptr>;

    using Transformation = kindr::minimal::QuatTransformation;

    inline static std::vector<dvs_msgs::Event *>::iterator EventVecPtr_lower_bound(
        std::vector<dvs_msgs::Event *> &vEventPtr, ros::Time &t)
    {
      return std::lower_bound(vEventPtr.begin(), vEventPtr.end(), t,
                              [](const dvs_msgs::Event *e, const ros::Time &t) { return e->ts.toSec() < t.toSec(); });
    }

    using EventQueue = std::deque<dvs_msgs::Event>;
    inline static EventQueue::iterator EventBuffer_lower_bound(
        EventQueue &eb, ros::Time &t)
    {
      return std::lower_bound(eb.begin(), eb.end(), t,
                              [](const dvs_msgs::Event &e, const ros::Time &t) { return e.ts.toSec() < t.toSec(); });
    }

    inline static EventQueue::iterator EventBuffer_upper_bound(
        EventQueue &eb, ros::Time &t)
    {
      return std::upper_bound(eb.begin(), eb.end(), t,
                              [](const ros::Time &t, const dvs_msgs::Event &e) { return t.toSec() < e.ts.toSec(); });
    }

    using StampTransformationMap = std::map<ros::Time, tools::Transformation>;
    inline static StampTransformationMap::iterator StampTransformationMap_lower_bound(
        StampTransformationMap &stm, ros::Time &t)
    {
      return std::lower_bound(stm.begin(), stm.end(), t,
                              [](const std::pair<ros::Time, tools::Transformation> &st, const ros::Time &t) { return st.first.toSec() < t.toSec(); });
    }

    /******************* Used by Block Match ********************/
    static inline void meanStdDev(
        Eigen::MatrixXd &patch,
        double &mean, double &sigma)
    {
      size_t numElement = patch.rows() * patch.cols();
      mean = patch.array().sum() / numElement;
      Eigen::MatrixXd sub = patch.array() - mean;
      sigma = sqrt((sub.array() * sub.array()).sum() / numElement) + 1e-6;
    }

    static inline void normalizePatch(
        Eigen::MatrixXd &patch_src,
        Eigen::MatrixXd &patch_dst)
    {
      double mean = 0;
      double sigma = 0;
      meanStdDev(patch_src, mean, sigma);
      patch_dst = (patch_src.array() - mean) / sigma;
    }

    // recursively create a directory
    static inline void _mkdir(const char *dir)
    {
      char tmp[256];
      char *p = NULL;
      size_t len;

      snprintf(tmp, sizeof(tmp), "%s", dir);
      len = strlen(tmp);
      if (tmp[len - 1] == '/')
        tmp[len - 1] = 0;
      for (p = tmp + 1; *p; p++)
        if (*p == '/')
        {
          *p = 0;
          mkdir(tmp, S_IRWXU);
          *p = '/';
        }
      mkdir(tmp, S_IRWXU);
    }

    // transform points
    template <typename PointType>
    inline void TransformPoint(const PointType pi, PointType &po, const Eigen::Matrix4d &T)
    {
      if (!pcl::traits::has_field<PointType, pcl::fields::intensity>::value)
      {
        std::cerr << "[TransformPoint] Point does not have intensity field!" << std::endl;
        exit(EXIT_FAILURE);
      }
      po = pi;
      Eigen::Vector3d point(pi.x, pi.y, pi.z);
      Eigen::Vector3d un_point = T.topLeftCorner<3, 3>() * point + T.topRightCorner<3, 1>();

      po.x = un_point.x();
      po.y = un_point.y();
      po.z = un_point.z();
      po.intensity = pi.intensity;
    }

  } // namespace tools
} // namespace esvo_core
#endif //ESVO_CORE_TOOLS_UTILS_H
