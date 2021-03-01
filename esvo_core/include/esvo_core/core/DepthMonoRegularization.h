#ifndef ESVO_CORE_CORE_DEPTHMONOREGULARIZATION_H
#define ESVO_CORE_CORE_DEPTHMONOREGULARIZATION_H

#include <esvo_core/container/DepthMap.h>
#include <esvo_core/core/DepthProblemConfig.h>
#include <memory>
namespace esvo_core
{
  using namespace container;
  namespace core
  {
    class DepthMonoRegularization
    {
    public:
      typedef std::shared_ptr<DepthMonoRegularization> Ptr;

      DepthMonoRegularization(std::shared_ptr<DepthProblemConfig> &dpConfigPtr);
      virtual ~DepthMonoRegularization();

      void apply(DepthMap::Ptr &depthMapPtr);
      void applyEdgeBased(DepthMap::Ptr &depthMapPtr);

    private:
      std::shared_ptr<DepthProblemConfig> dpConfigPtr_;
      size_t _regularizationRadius;
      size_t _regularizationMinNeighbours;
      size_t _regularizationMinCloseNeighbours;
    };
  } // namespace core
} // namespace esvo_core

#endif //ESVO_CORE_CORE_DepthMonoRegularization_H