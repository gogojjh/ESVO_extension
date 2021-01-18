#ifndef MAPPER_EMVS_H_
#define MAPPER_EMVS_H_

#include <dvs_msgs/Event.h>

#include <opencv2/opencv.hpp>

#include <eigen3/Eigen/Dense>

#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>

#include "esvo_core/container/DepthPoint.h"
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "cartesian3dgrid/cartesian3dgrid.h"
#include "Trajectory.hpp"
#include "DepthVector.hpp"

#define USE_INVERSE_DEPTH

namespace EMVS
{
#ifdef USE_INVERSE_DEPTH
	using TypeDepthVector = InverseDepthVector;
#else
	using TypeDepthVector = LinearDepthVector;
#endif

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
			std::cout << "depth range: " << min_depth_ << " -> " << max_depth_ << std::endl;
			std::cout << "fov: " << fov_ << std::endl;
			std::cout << "**********************************" << std::endl;
		}
	};

	struct OptionsDepthMap
	{
		// Adaptive Gaussian Thresholding parameters
		int adaptive_threshold_kernel_size_;
		double adaptive_threshold_c_;

		// Kernel size of median filter
		int median_filter_size_;
	};

	struct OptionsPointCloud
	{
		// Outlier removal parameters
		float radius_search_;
		int min_num_neighbors_;
	};

	class MapperEMVS
	{
	public:
		MapperEMVS() {}

		MapperEMVS(const camodocal::CameraPtr &camera_ptr,
				   const camodocal::CameraPtr &camera_virtual_ptr);

		void configDSI(ShapeDSI &dsi_shape);

		void initializeDSI(const Eigen::Matrix4d &T_w_rv);

		bool updateDSI(const std::vector<dvs_msgs::Event> &events,
					   const LinearTrajectory &trajectory,
					   const PoseMap &poses);

		void getDepthMapFromDSI(cv::Mat &depth_map, cv::Mat &confidence_map, cv::Mat &mask, const OptionsDepthMap &options_depth_map);

		void getDepthPoint(const cv::Mat &depth_map,
						   const cv::Mat &mask,
						   std::vector<esvo_core::container::DepthPoint> &vdp);

		void getPointcloud(const cv::Mat &depth_map,
						   const cv::Mat &mask,
						   const OptionsPointCloud &options_pc,
						   pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_);

		Grid3D dsi_;

		Eigen::Matrix4d T_w_rv_;

	private:
		void precomputeRectifiedPoints();
		void fillVoxelGrid(const std::vector<Eigen::Vector4f> &event_locations_z0,
						   const std::vector<Eigen::Vector3f> &camera_centers);
		void convertDepthIndicesToValues(const cv::Mat &depth_cell_indices, cv::Mat &depth_map);
		void removeMaskBoundary(cv::Mat &mask, int border_size);

		// Intrinsics of the camera
		camodocal::CameraPtr camera_ptr_;
		camodocal::CameraPtr camera_virtual_ptr_;

		std::vector<double> camera_params_;
		std::vector<double> camera_virtual_params_;

		Eigen::Matrix3f K_;
		int width_;
		int height_;

		// (Constant) parameters that define the DSI (size and intrinsics)
		ShapeDSI dsi_shape_;

		// Precomputed vector of num_depth_cells_ inverse depths,
		// uniformly sampled in inverse depth space
		TypeDepthVector depths_vec_;
		std::vector<float> raw_depths_vec_;

		// Precomputed (normalized) bearing vectors for each pixel of the reference image
		Eigen::Matrix2Xf precomputed_rectified_points_;

		const size_t packet_size_ = 1024;
	};

} // namespace EMVS

#endif