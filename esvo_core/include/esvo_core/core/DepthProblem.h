#ifndef ESVO_CORE_CORE_DEPTHPROBLEM_H
#define ESVO_CORE_CORE_DEPTHPROBLEM_H

#include <esvo_core/tools/utils.h>
#include <esvo_core/core/DepthProblemConfig.h>
#include <esvo_core/container/CameraSystem.h>
#include <esvo_core/container/TimeSurfaceObservation.h>
#include <esvo_core/optimization/OptimizationFunctor.h>

namespace esvo_core
{
  using namespace container;
  using namespace tools;
  namespace core
  {
    struct DepthProblem : public optimization::OptimizationFunctor<double>
    {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      DepthProblem(
          const DepthProblemConfig::Ptr &dpConfig_ptr,
          const CameraSystem::Ptr &camSysPtr);

      void setProblem(
          Eigen::Vector2d &coor,
          Eigen::Matrix<double, 4, 4> &T_world_virtual,
          StampedTimeSurfaceObs *pStampedTsObs);

      // function that is inherited from optimization::OptimizationFunctor
      int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const;

      // utils
      bool warping(
          const Eigen::Vector2d &x,
          double d,
          const Eigen::Matrix<double, 3, 4> &T_left_virtual,
          Eigen::Vector2d &x1_s,
          Eigen::Vector2d &x2_s) const;

      bool patchInterpolation(
          const Eigen::MatrixXd &img,
          const Eigen::Vector2d &location,
          Eigen::MatrixXd &patch,
          bool debug = false) const;

      // variables
      CameraSystem::Ptr camSysPtr_;
      DepthProblemConfig::Ptr dpConfigPtr_;
      Eigen::Vector2d coordinate_;
      Eigen::Matrix<double, 4, 4> T_world_virtual_;
      std::vector<Eigen::Matrix<double, 3, 4>,
                  Eigen::aligned_allocator<Eigen::Matrix<double, 3, 4>>>
          vT_left_virtual_;
      StampedTimeSurfaceObs *pStampedTsObs_;
    };
  } // namespace core
} // namespace esvo_core

#endif //ESVO_CORE_CORE_DEPTHPROBLEM_H
