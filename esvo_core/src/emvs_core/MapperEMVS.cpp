#include <pcl/filters/radius_outlier_removal.h>

#include "emvs_core/MapperEMVS.hpp"
#include "emvs_core/MedianFilter.hpp"

namespace EMVS
{
	MapperEMVS::MapperEMVS(const camodocal::CameraPtr &camera_ptr,
						   const camodocal::CameraPtr &camera_virtual_ptr)
		: camera_ptr_(camera_ptr),
		  camera_virtual_ptr_(camera_virtual_ptr)
	{
		width_ = camera_ptr_->imageWidth();
		height_ = camera_ptr_->imageHeight();
		camera_ptr_->writeParameters(camera_params_);
		precomputeRectifiedPoints();
	}

	void MapperEMVS::configDSI(ShapeDSI &dsi_shape)
	{
		CHECK_GT(dsi_shape.min_depth_, 0.0);
		CHECK_GT(dsi_shape.max_depth_, dsi_shape.min_depth_);
		depths_vec_ = TypeDepthVector(float(dsi_shape.min_depth_), float(dsi_shape.max_depth_), float(dsi_shape.dimZ_));
		raw_depths_vec_ = depths_vec_.getDepthVector();
		dsi_shape.dimX_ = (dsi_shape.dimX_ > 0) ? dsi_shape.dimX_ : camera_ptr_->imageWidth();
		dsi_shape.dimY_ = (dsi_shape.dimY_ > 0) ? dsi_shape.dimY_ : camera_ptr_->imageHeight();
		dsi_shape.fov_ = (dsi_shape.fov_ > 0) ? dsi_shape.fov_ : 45.0; // using event camera's fov
		camera_virtual_params_ = std::vector<double>{0, 0, 0, 0,
													 camera_params_[4],
													 camera_params_[4],
													 0.5 * dsi_shape.dimX_,
													 0.5 * dsi_shape.dimY_};
		camera_virtual_ptr_->readParameters(camera_virtual_params_);
		K_virtual_ << camera_virtual_params_[4], 0.0, camera_virtual_params_[6],
   			          0.0, camera_virtual_params_[5], camera_virtual_params_[7],
   			          0.0, 0.0, 1.0;
		dsi_ = Grid3D(dsi_shape.dimX_, dsi_shape.dimY_, dsi_shape.dimZ_);
	}

	void MapperEMVS::initializeDSI(const Eigen::Matrix4d &T_w_rv)
	{
		T_w_rv_ = T_w_rv;
		dsi_.resetGrid();
	}

	bool MapperEMVS::updateDSI(const std::map<ros::Time, Eigen::Matrix4d> pVirtualPoses,
  							   const std::vector<dvs_msgs::Event *> pvEventsPtr)
	{
		CHECK_GT(pVirtualPoses.size(), 1);

		// 2D coordinates of the events transferred to reference view using plane Z = Z_0.
		// We use Vector4f because Eigen is optimized for matrix multiplications with inputs whose size is a multiple of 4
		std::vector<Eigen::Vector4f> event_locations_z0;
		event_locations_z0.clear();
		event_locations_z0.reserve(pvEventsPtr.size());
	
		// List of camera centers
		std::vector<Eigen::Vector3f> camera_centers;
		camera_centers.clear();
		camera_centers.reserve(pvEventsPtr.size());

		auto it_ev_begin = pvEventsPtr.begin();
		for (auto it_vp = pVirtualPoses.begin(); it_vp != pVirtualPoses.end(); it_vp++)
		{
			Eigen::Matrix4f T_w_ev = it_vp->second.cast<float>();
			Eigen::Matrix4f T_ev_rv = T_w_ev.inverse() * T_w_rv_.cast<float>();
			Eigen::Matrix3f R = T_ev_rv.topLeftCorner<3, 3>();
			Eigen::Vector3f t = T_ev_rv.topRightCorner<3, 1>();
			Eigen::Vector3f c = -R.transpose() * t;

			// Project the points on plane at distance z0
			const float z0 = raw_depths_vec_[0];
			// Planar homography  (H_z0)^-1 that maps a point in the reference view to the event camera through plane Z = Z0 (Eq. (8) in the IJCV paper)
			// Planar homography  (H_z0) transforms [u, v] to [X(Z0), Y(Z0), 1]
			Eigen::Matrix3f H_z0_inv = R;
			H_z0_inv *= z0;
			H_z0_inv.col(2) += t;
			// Compute H_z0 in pixel coordinates using the intrinsic parameters
			Eigen::Matrix3f H_z0_px = K_virtual_.cast<float>() * H_z0_inv.inverse(); // transform [u, v] to [X(Z0), Y(Z0), 1]

			// Use a 4x4 matrix to allow Eigen to optimize the speed
			Eigen::Matrix4f H_z0_px_4x4;
			H_z0_px_4x4.block<3, 3>(0, 0) = H_z0_px;
			H_z0_px_4x4.col(3).setZero();
			H_z0_px_4x4.row(3).setZero();

			for (auto it_ev = it_ev_begin; it_ev != pvEventsPtr.end(); it_ev++)
			{
				if ((*it_ev)->ts.toSec() > it_vp->first.toSec())
				{
					it_ev_begin = it_ev;
					break;
				}

				if ((*it_ev)->y * width_ + (*it_ev)->x < 0 || 
					(*it_ev)->y * width_ + (*it_ev)->x >= precomputed_rectified_points_.cols())
					continue;

				// For each event, precompute the warped event locations according to Eq. (11) in the IJCV paper.
				Eigen::Vector4f p;
				p.head<2>() = precomputed_rectified_points_.col((*it_ev)->y * width_ + (*it_ev)->x);
				p[2] = 1.;
				p[3] = 0.;
				p = H_z0_px_4x4 * p;
				p /= p[2];
				event_locations_z0.push_back(p);
				// Optical center of the event camera in the coordinate frame of the reference view
				camera_centers.push_back(c);
			}
			// LOG(INFO) << "number of virtual view: " << camera_centers.size();
		}

		fillVoxelGrid(event_locations_z0, camera_centers);
		return true;
	}

		void MapperEMVS::fillVoxelGrid(const std::vector<Eigen::Vector4f> &event_locations_z0,
									   const std::vector<Eigen::Vector3f> &camera_centers)
		{
			CHECK_EQ(event_locations_z0.size(), camera_centers.size());

			// This function implements Step 2 of Algorithm 1 in the IJCV paper.
			// It maps events from plane Z0 to all the planes Zi of the DSI using Eq. (15)
			// and then votes for the corresponding voxel using bilinear voting.

			// For efficiency reasons, we split each packet into batches of N events each
			// which allows to better exploit the L1 cache
			// static const int N = 128;
			// typedef Eigen::Array<float, N, 1> Arrayf;
			const float z0 = raw_depths_vec_[0];

			// Parallelize over the planes of the DSI with OpenMP
			// (each thread will process a different depth plane)
			#pragma omp parallel for if (event_locations_z0.size() >= 20000)
			for (size_t depth_plane = 0; depth_plane < raw_depths_vec_.size(); ++depth_plane)
			{
				float *pgrid = dsi_.getPointerToSlice(depth_plane); // the 2D cells
				for (size_t i = 0; i < camera_centers.size(); i++)
				{
					// Precompute coefficients for Eq. (15)
					const Eigen::Vector3f &C = camera_centers[i];
					const float zi = static_cast<float>(raw_depths_vec_[depth_plane]);
					const float a = z0 * (zi - C[2]);
					const float bx = (z0 - zi) * (C[0] * camera_virtual_params_[4] + C[2] * camera_virtual_params_[6]);
					const float by = (z0 - zi) * (C[1] * camera_virtual_params_[5] + C[2] * camera_virtual_params_[7]);
					const float d = zi * (z0 - C[2]);

					// Eq. (15)
					// Arrayf X, Y;
					// for (size_t i = 0; i < N; ++i)
					// {
					// 	X[i] = pe[i][0];
					// 	Y[i] = pe[i][1];
					// }
					float X = (event_locations_z0[i][0] * a + bx) / d;
					float Y = (event_locations_z0[i][1] * a + by) / d;
					// Bilinear voting
					dsi_.accumulateGridValueAt(X, Y, pgrid);
				}
			}
		}

		void MapperEMVS::precomputeRectifiedPoints()
		{
			// Create a lookup table that maps pixel coordinates to undistorted pixel coordinates
			precomputed_rectified_points_ = Eigen::Matrix2Xf(2, height_ * width_);
			for (int y = 0; y < height_; y++)
			{
				for (int x = 0; x < width_; ++x)
				{
					Eigen::Vector2d p_d(x, y);
					Eigen::Vector3d P_u;

					camera_ptr_->liftProjective(p_d, P_u);
					P_u /= P_u.z();
					precomputed_rectified_points_.col(y * width_ + x) = P_u.head<2>().cast<float>(); // lift a point on its projective ray
				}
			}
		}

		void MapperEMVS::convertDepthIndicesToValues(const cv::Mat &depth_cell_indices, cv::Mat &depth_map)
		{
			// Convert depth indices to depth values, for all pixels
			depth_map = cv::Mat(depth_cell_indices.rows, depth_cell_indices.cols, CV_32F);
			for (int y = 0; y < depth_cell_indices.rows; ++y)
			{
				for (int x = 0; x < depth_cell_indices.cols; ++x)
				{
					depth_map.at<float>(y, x) = depths_vec_.cellIndexToDepth(depth_cell_indices.at<uchar>(y, x));
				}
			}
		}

		void MapperEMVS::removeMaskBoundary(cv::Mat & mask, int border_size)
		{
			for (int y = 0; y < mask.rows; ++y)
			{
				for (int x = 0; x < mask.cols; ++x)
				{
					if (x <= border_size || x >= mask.cols - border_size ||
						y <= border_size || y >= mask.rows - border_size)
					{
						mask.at<uchar>(y, x) = 0;
					}
				}
			}
		}

		void MapperEMVS::getDepthMapFromDSI(cv::Mat & depth_map, cv::Mat & confidence_map, cv::Mat & mask, const OptionsDepthMap &options_depth_map)
		{
			// Reference: Section 5.2.3 in the IJCV paper.

			// Maximum number of votes along optical ray
			cv::Mat depth_cell_indices;
			dsi_.collapseMaxZSlice(&confidence_map, &depth_cell_indices);

			// Adaptive thresholding on the confidence map
			cv::Mat confidence_8bit;
			cv::normalize(confidence_map, confidence_8bit, 0.0, 255.0, cv::NORM_MINMAX);
			confidence_8bit.convertTo(confidence_8bit, CV_8U);
			cv::adaptiveThreshold(confidence_8bit,
								  mask,
								  1,
								  cv::ADAPTIVE_THRESH_GAUSSIAN_C,
								  cv::THRESH_BINARY,
								  options_depth_map.adaptive_threshold_kernel_size_,
								  -options_depth_map.adaptive_threshold_c_);

			// Clean up depth map using median filter (Section 5.2.5 in the IJCV paper)
			cv::Mat depth_cell_indices_filtered;
			huangMedianFilter(depth_cell_indices,
							  depth_cell_indices_filtered,
							  mask,
							  options_depth_map.median_filter_size_);

			// Remove the outer border to suppress boundary effects
			const int border_size = std::max(options_depth_map.adaptive_threshold_kernel_size_ / 2, 1);
			removeMaskBoundary(mask, border_size);

			// Convert depth indices to depth values
			convertDepthIndicesToValues(depth_cell_indices_filtered, depth_map);
		}

		void MapperEMVS::getDepthPoint(const cv::Mat &depth_map,
									   const cv::Mat &mask,
									   std::vector<esvo_core::container::DepthPoint> &vdp)
		{
			vdp.clear();
			for (size_t y = 0; y < depth_map.rows; ++y)
			{
				for (size_t x = 0; x < depth_map.cols; ++x)
				{
					if (mask.at<uint8_t>(y, x) > 0)
					{
						Eigen::Vector2d p(x, y);
						Eigen::Vector3d P;
						camera_virtual_ptr_->liftProjective(p, P);
						Eigen::Vector3d xyz_rv = (P / P.z() * depth_map.at<float>(y, x));
						if (xyz_rv.z() <= 1e-6)
							continue;

						double variance = 0.1;
						esvo_core::container::DepthPoint dp(y, x);
						dp.update_x(p);
						dp.update(1.0 / xyz_rv.z(), variance);
						dp.update_p_cam(xyz_rv);
						dp.updatePose(T_w_rv_);
						vdp.push_back(dp);
					}
				}
			}
		}

		void MapperEMVS::getPointcloud(const cv::Mat &depth_map,
									   const cv::Mat &mask,
									   const OptionsPointCloud &options_pc,
									   pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_)
		{
			CHECK_EQ(depth_map.rows, mask.rows);
			CHECK_EQ(depth_map.cols, mask.cols);

			// Convert depth map to point cloud
			pc_->clear();
			for (int y = 0; y < depth_map.rows; ++y)
			{
				for (int x = 0; x < depth_map.cols; ++x)
				{
					if (mask.at<uint8_t>(y, x) > 0)
					{
						Eigen::Vector2d p(x, y);
						Eigen::Vector3d P;
						camera_virtual_ptr_->liftProjective(p, P);
						Eigen::Vector3d xyz_rv = (P / P.z() * depth_map.at<float>(y, x));

						pcl::PointXYZI p_rv; // 3D point in reference view
						p_rv.x = xyz_rv.x();
						p_rv.y = xyz_rv.y();
						p_rv.z = xyz_rv.z();
						p_rv.intensity = 1.0 / p_rv.z;
						pc_->push_back(p_rv);
					}
				}
			}

			// Filter point cloud to remove outliers (Section 5.2.5 in the IJCV paper)
			pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);
			pcl::RadiusOutlierRemoval<pcl::PointXYZI> outlier_rm;
			outlier_rm.setInputCloud(pc_);
			outlier_rm.setRadiusSearch(options_pc.radius_search_);
			outlier_rm.setMinNeighborsInRadius(options_pc.min_num_neighbors_);
			outlier_rm.filter(*cloud_filtered);
			pc_->swap(*cloud_filtered);
		}

	} // namespace EMVS
