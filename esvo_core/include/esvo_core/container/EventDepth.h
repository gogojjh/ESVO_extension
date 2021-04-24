#ifndef ESVO_CORE_CORE_EVENTDEPTH_H
#define ESVO_CORE_CORE_EVENTDEPTH_H

#include <vector>
#include <esvo_core/tools/utils.h>
#include <esvo_core/container/CameraSystem.h>
#include <esvo_core/container/TimeSurfaceObservation.h>
#include <deque>

namespace esvo_core
{
using namespace container;
using namespace tools;
namespace core
{
struct EventDepth
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EventDepth() {}

  // raw event coordinate
  Eigen::Vector2d x_raw_;
  // rectified_event coordinate (left, right)
  Eigen::Vector2d x_rect_;
  // timestamp
  ros::Time t_;
  // pose of virtual view (T_world_virtual)
  Transformation trans_;
  // inverse depth
  double depth_;
  // spatio-temporal cost
  double cost_;
  // status: 0: no LiDAR depth; 1: have LiDAR depth
  int status_;
};
}
}

#endif //ESVO_CORE_CORE_EVENTDEPTH_H
