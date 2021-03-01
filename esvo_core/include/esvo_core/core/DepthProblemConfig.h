#ifndef ESVO_CORE_CORE_DEPTHPROBLEM_CONFIG_H
#define ESVO_CORE_CORE_DEPTHPROBLEM_CONFIG_H

namespace esvo_core
{
  namespace core
  {
    struct DepthProblemConfig
    {
      using Ptr = std::shared_ptr<DepthProblemConfig>;
      DepthProblemConfig(
          size_t patchSize_X,
          size_t patchSize_Y,
          const std::string &LSnorm,
          double td_nu,
          double td_scale,
          const size_t MAX_ITERATION = 10)
          : patchSize_X_(patchSize_X),
            patchSize_Y_(patchSize_Y),
            LSnorm_(LSnorm),
            Noise_Model_(std::string("Gaussian")),
            td_nu_(td_nu),
            td_scale_(td_scale),
            td_scaleSquared_(pow(td_scale, 2)),
            td_stdvar_(sqrt(td_nu / (td_nu - 2) * td_scaleSquared_)),
            MAX_ITERATION_(MAX_ITERATION) {}

      size_t patchSize_X_, patchSize_Y_;
      std::string LSnorm_;
      std::string Noise_Model_;
      double td_nu_;
      double td_scale_;
      double td_scaleSquared_; // td_scale_^2
      double td_stdvar_;       // sigma
      size_t MAX_ITERATION_;
    };
  } // namespace core
} // namespace esvo_core

#endif