#ifndef TRAJECTORY_HPP_
#define TRAJECTORY_HPP_

#include <glog/logging.h>

#include <map>
#include <algorithm>
#include <iostream>
#include <ros/time.h>
#include <eigen3/Eigen/Dense>

// #define TRAJECTORY_LOG

namespace EMVS
{
	typedef std::map<ros::Time, Eigen::Matrix4d> PoseMap;

	// return the first iterator which is greater than t
	inline PoseMap::const_iterator PoseMap_lower_bound(
		const PoseMap &stm, const ros::Time &t)
	{
		return std::lower_bound(stm.begin(), stm.end(), t,
								[](const std::pair<ros::Time, Eigen::Matrix4d> &st, const ros::Time &t) { return st.first.toSec() < t.toSec(); });
	}

	template <class DerivedTrajectory>
	class Trajectory
	{
	public:
		DerivedTrajectory &derived()
		{
			return static_cast<DerivedTrajectory &>(*this);
		}

		Trajectory() {}

		// Returns T_W_C (mapping points from C to the world frame W)
		bool getPoseAt(const PoseMap &pose_map, const ros::Time &t, Eigen::Matrix4d &T) const
		{
			return derived().getPoseAt(t, T);
		}

		void getFirstControlPose(const PoseMap &pose_map, Eigen::Matrix4d *T, ros::Time *t) const
		{
			*t = pose_map.begin()->first;
			*T = pose_map.begin()->second;
		}

		void getLastControlPose(const PoseMap &pose_map, Eigen::Matrix4d *T, ros::Time *t) const
		{
			*t = pose_map.rbegin()->first;
			*T = pose_map.rbegin()->second;
		}

		size_t getNumControlPoses(const PoseMap &pose_map) const
		{
			return pose_map.size();
		}

		bool printAllPoses(const PoseMap &pose_map) const
		{
			size_t control_pose_idx = 0u;
			for (auto it : pose_map)
			{
				std::cout << "--Control pose #" << control_pose_idx++ << ". time = " << it.first << std::endl;
				std::cout << "--T = " << std::endl;
				std::cout << it.second << std::endl;
			}
			return true;
		}
	};

	class LinearTrajectory : public Trajectory<LinearTrajectory>
	{
	public:
		LinearTrajectory() : Trajectory() {}

		bool getPoseAt(const PoseMap &pose_map, const ros::Time &t, Eigen::Matrix4d &T) const
		{
			ros::Time t0_, t1_;
			Eigen::Matrix4d T0_, T1_;

			// Check if it is between two known poses
			auto it1 = PoseMap_lower_bound(pose_map, t);
			if (it1 == pose_map.begin())
			{
#ifdef TRAJECTORY_LOG
				std::cout << "Cannot extrapolate in the past. Requested pose: "
						  << t << " but the earliest pose available is at time: "
						  << pose_map.begin()->first << std::endl;
#endif
				return false;
			}
			else if (it1 == pose_map.end())
			{
#ifdef TRAJECTORY_LOG
				std::cout << "Cannot extrapolate in the future. Requested pose: "
						  << t << " but the latest pose available is at time: "
						  << (pose_map.rbegin())->first << std::endl
#endif
				return false;
			}
			else
			{
				auto it0 = std::prev(it1);
				t0_ = (it0)->first;
				T0_ = (it0)->second;
				t1_ = (it1)->first;
				T1_ = (it1)->second;
#ifdef TRAJECTORY_LOG
				printf("interpolation: %f < %f < %f\n\n", it0->first.toSec(), t.toSec(), it1->first.toSec());
#endif
			}

			// Linear interpolation in SE(3)
			const Eigen::Matrix4d T_relative = T0_.inverse() * T1_;
			const double delta_t = (t - t0_).toSec() / (t1_ - t0_).toSec();
			T = T0_ * getLinearInterpolation(T_relative, delta_t);
			return true;
		}

		Eigen::Matrix4d getLinearInterpolation(const Eigen::Matrix4d &T_relative, const double &delta_t) const
		{
			Eigen::Matrix3d R_relative = T_relative.topLeftCorner<3, 3>();
			Eigen::Quaterniond q_relative(R_relative);

			Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
			T.topLeftCorner<3, 3>() = q_relative.slerp(delta_t, q_relative).toRotationMatrix();
			T.topRightCorner<3, 1>() = delta_t * T_relative.topRightCorner<3, 1>();
			return T;
		}
	};
} // namespace EMVS

#endif