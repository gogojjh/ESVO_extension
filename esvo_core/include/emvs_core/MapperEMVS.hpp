// Implementation of plane sweep (stereo) methods on event camera-based mapping

#ifndef MAPPER_EMVS_H_
#define MAPPER_EMVS_H_

#include <dvs_msgs/Event.h>

#include <eigen3/Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>

#include "esvo_core/container/DepthPoint.h"
#include "esvo_core/container/CameraSystem.h"
#include "cartesian3dgrid/cartesian3dgrid.h"
#include "DepthVector.hpp"

#define USE_INVERSE_DEPTH
// #define EMVS_CHECK_OBS

namespace EMVS
{
#ifdef USE_INVERSE_DEPTH
	using TypeDepthVector = InverseDepthVector;
#else
	using TypeDepthVector = LinearDepthVector;
#endif

	// (Constant) parameters that define the DSI (size and intrinsics)
	struct ShapeDSI
	{
	public:
		ShapeDSI() {}
		ShapeDSI(size_t dimX, size_t dimY, size_t dimZ,
				 double min_depth, double max_depth, double fov) : dimX_(dimX),
																   dimY_(dimY),
																   dimZ_(dimZ),
																   min_depth_(min_depth),
																   max_depth_(max_depth),
																   fov_(fov)
		{}

		size_t dimX_;
		size_t dimY_;
		size_t dimZ_;

		double min_depth_;
		double max_depth_;
		double fov_; // Field of View

		inline void printDSIInfo()
		{
			std::cout << "************ DSI info ************" << std::endl;
			std::cout << "dimension (XYZ): " << dimX_ << ", " << dimY_ << ", " << dimZ_ << std::endl;
			std::cout << "depth range [m]: " << min_depth_ << " -> " << max_depth_ << std::endl;
			std::cout << "fov: " << fov_ << std::endl;
			std::cout << "**********************************" << std::endl;
		}
	};

	struct OptionsDepthMap
	{
	public:
		OptionsDepthMap(const int &adaptive_threshold_kernel_size,
						const double &adaptive_threshold_c,
						const int &median_filter_size,
						const int &contrast_threshold) : adaptive_threshold_kernel_size_(adaptive_threshold_kernel_size),
														 adaptive_threshold_c_(adaptive_threshold_c),
														 median_filter_size_(median_filter_size),
														 contrast_threshold_(contrast_threshold)
		{
		}

		// Adaptive Gaussian Thresholding parameters
		int adaptive_threshold_kernel_size_;
		double adaptive_threshold_c_;
		// Kernel size of median filter
		int median_filter_size_;
		// contrast Thresholding parameters to remove outliers
		int contrast_threshold_;
	};

	struct OptionsPointCloud
	{
	public:
		OptionsPointCloud(const float &radius_search, const int &min_num_neighbors) : radius_search_(radius_search),
																					  min_num_neighbors_(min_num_neighbors)
		{

		}

		// Outlier removal parameters
		float radius_search_;
		int min_num_neighbors_;
	};

	struct OptionsMapper
	{
	public:
		OptionsMapper(const float &min_parallex, const int &obs_patch_size_x, const int &obs_patch_size_y, const float &min_ts_score) : min_parallex_(min_parallex),
																																		obs_patch_size_x_(obs_patch_size_x),
																																		obs_patch_size_y_(obs_patch_size_y),
																																		min_ts_score_(min_ts_score)
		{

		}
		float min_parallex_;
		int obs_patch_size_x_, obs_patch_size_y_;
		float min_ts_score_;
	};

	class MapperEMVS
	{
	public:
		MapperEMVS() {}
		MapperEMVS(const esvo_core::container::PerspectiveCamera::Ptr &camPtr, ShapeDSI &dsi_shape, const OptionsMapper &opts_mapper);

		void reset();

		void initializeDSI(const Eigen::Matrix4d &T_w_rv);
		bool updateDSI();

		void getDepthMapFromDSI(cv::Mat &depth_map,
								cv::Mat &confidence_map,
								cv::Mat &mask,
								const OptionsDepthMap &options_depth_map,
								double &mean_depth);

		void getProbMapFromDSI(cv::Mat &mean_map, cv::Mat &variance_map);

		void getDepthPoint(const cv::Mat &depth_map,
						   const cv::Mat &confidence_map,
						   const cv::Mat &mask,
						   std::vector<esvo_core::container::DepthPoint> &vdp,
						   const double &stdVar_init);

		void getPointcloud(const cv::Mat &depth_map,
						   const cv::Mat &mask,
						   const OptionsPointCloud &options_pc,
						   pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_);

		void storeEventsPose(std::vector<std::pair<ros::Time, Eigen::Matrix4d>> &pVirtualPoses,
							 std::vector<Eigen::Vector4d> &pvEventsPtr);

		void setTSNegativeObservation(std::shared_ptr<Eigen::MatrixXd> &pTSNegative)
		{
			pTSNegative_ = pTSNegative;
		}

		void computeObservation(const int &num_event);

		inline size_t storeEventNum()
		{
			return vpEventsPose_.size();
		}

		inline void clearEvents()
		{
			vpEventsPose_.clear();
			vpEventsPose_.reserve(1e5);
		}

		inline ros::Time getRVTime()
		{
			return ros::Time((*vpEventsPose_[vpEventsPose_.size() / 2].first)[2]);
		}

		Grid3D dsi_;
		Eigen::Matrix4d T_w_rv_;
		bool dsiInitFlag_;
		size_t accu_event_number_;

	private:
		void precomputeRectifiedPoints();
		void fillVoxelGrid(const std::vector<Eigen::Vector4f> &event_locations_z0,
						   const std::vector<Eigen::Vector3f> &camera_centers);
		bool observedTS(const float &x, const float &y);
		void convertDepthIndicesToValues(const cv::Mat &depth_cell_indices, cv::Mat &depth_map);
		void removeMaskBoundary(cv::Mat &mask, int border_size);

		Eigen::Matrix3f K_virtual_, K_;
		int width_;
		int height_;

		// Precomputed vector of num_depth_cells_ inverse depths,
		// uniformly sampled in inverse depth space
		TypeDepthVector depths_vec_;
		std::vector<float> raw_depths_vec_;

		// Precomputed (normalized) bearing vectors for each pixel of the reference image
		Eigen::Matrix2Xf precomputed_rectified_points_;

		std::vector<std::pair<std::shared_ptr<Eigen::Vector4d>, std::shared_ptr<Eigen::Matrix4d>>> vpEventsPose_;
		std::shared_ptr<Eigen::MatrixXd> pTSNegative_;

		float min_Parallax_;
		int obs_PatchSize_X_;
		int obs_PatchSize_Y_;
		float min_TS_Score_;
	};

} // namespace EMVS

#endif