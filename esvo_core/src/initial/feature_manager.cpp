/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <initial/feature_manager.h>

using namespace std;
using namespace Eigen;

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager()
{
}

void FeatureManager::clearState()
{
    feature_.clear();
}

int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature_)
    {
        it.used_num = it.feature_per_frame.size();
        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;
        }
    }
    return cnt;
}

bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    // LOG(INFO) << "input feature: " << image.size();
    last_track_num_ = 0;
    last_average_parallax_ = 0;
    new_feature_num_ = 0;
    long_track_num_ = 0;

    // add features from the new frame
    for (auto &id_pts : image)
    {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td); // id_pts.second[0].second: the feature vector of the first camera
        int feature_id = id_pts.first;
        auto it = find_if(feature_.begin(), feature_.end(), [feature_id](const FeaturePerId &it) 
        {
            return it.feature_id == feature_id;
        });

        if (it == feature_.end())
        {
            feature_.push_back(FeaturePerId(feature_id, frame_count)); // feature_id and start_frame
            feature_.back().feature_per_frame.push_back(f_per_fra);
            // new_feature_num_++;
        }
        else if (it->feature_id == feature_id)
        {
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num_++;
        }
    }

    if (frame_count < 2 || last_track_num_ < 20)
        return true;

    // keyframe selection 2: average parallax
    double parallax_sum = 0;
    int parallax_num = 0;
    for (auto &it_per_id : feature_)
    {
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }

    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        // printf("parallax_sum: %lf, parallax_num: %d\n", parallax_sum, parallax_num);
        // printf("current parallax: %lf\n", parallax_sum / parallax_num * FOCAL_LENGTH);
        // printf("average parallax: %lf\n", parallax_sum / parallax_num);
        last_average_parallax_ = parallax_sum / parallax_num * FOCAL_LENGTH;
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

// find correspondences between frame_count_l <-> frame_count_
vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature_)
    {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;
            a = it.feature_per_frame[idx_l].point;
            b = it.feature_per_frame[idx_r].point;
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature_)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = feature_.begin(), it_next = feature_.begin();
         it != feature_.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature_.erase(it);
    }
}

void FeatureManager::clearDepth()
{
    for (auto &it_per_id : feature_)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth = -1.0;
    }
}

VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature_)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth; // optimize the inverse depth
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

void FeatureManager::triangulate(Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[])
{
    for (auto &it_per_id : feature_)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        if (it_per_id.estimated_depth > 0)
            continue;
        int frame_i = it_per_id.start_frame;
        int frame_j = frame_i - 1;

        ROS_ASSERT(NUM_OF_CAM == 1);
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0;
        Eigen::Vector3d t0 = Ps[frame_i] + Rs[frame_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[frame_i] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            frame_j++;

            Eigen::Vector3d t1 = Ps[frame_j] + Rs[frame_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[frame_j] * ric[0];
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame.point.normalized();
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (frame_i == frame_j)
                continue;
        }
        ROS_ASSERT(svd_idx == svd_A.rows());
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];
        it_per_id.estimated_depth = svd_method;

        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = INIT_DEPTH;
        }
    }
}

void FeatureManager::removeOutlier(set<int> &outlierIndex)
{
    std::set<int>::iterator itSet;
    for (auto it = feature_.begin(), it_next = feature_.begin();
         it != feature_.end(); it = it_next)
    {
        it_next++;
        int index = it->feature_id;
        itSet = outlierIndex.find(index);
        if (itSet != outlierIndex.end())
        {
            feature_.erase(it);
            //printf("remove outlier %d \n", index);
        }
    }
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature_.begin(), it_next = feature_.begin();
         it != feature_.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() < 2)
            {
                feature_.erase(it);
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}

void FeatureManager::removeBack()
{
    for (auto it = feature_.begin(), it_next = feature_.begin();
         it != feature_.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0) // if feature does not start at the the oldest window
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin()); // if feature starts at the the oldest window
            if (it->feature_per_frame.size() == 0)
                feature_.erase(it);
        }
    }
}

void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature_.begin(), it_next = feature_.begin(); it != feature_.end(); it = it_next)
    {
        it_next++;
        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1)
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)
                feature_.erase(it);
        }
    }
}

// compute the parallax on the normalized image plane
// normalized image plane: P = [X/Z, Y/Z, 1]
double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax between seconde last frame and third last frame
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];
    
    Vector3d p_j = frame_j.point;
    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    Vector3d p_i_comp = p_i;
    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    double ans = 0;
    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));
    return ans;
}

// ----------------------------------------------------------------
// ------------------------------- modified by jjiao
// ----------------------------------------------------------------
void FeatureManager::initDepth()
{
    for (auto &it_per_id : feature_)
    {
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth = INIT_DEPTH;
    }
}

// TODO: check the correct triangulation
// void FeatureManager::triangulate(const CircularBuffer<Vector3d> &Ps, const CircularBuffer<Quaterniond> &Qs,
//                                  const Eigen::Vector3d &tbc, const Eigen::Quaterniond &qbc)
// {
//     int succ_tagl_cnt = 0;
//     for (auto &it_per_id : feature_)
//     {
//         it_per_id.used_num = it_per_id.feature_per_frame.size();

//         if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
//             continue;

//         if (it_per_id.estimated_depth > 0)
//             continue;

//         {
        // version of VINS-Fusion
        // observed by >= 2 frames
        // triangulate points using the first two frames
        // if (it_per_id.used_num >= 2)
        // {
        //     int frame_i = it_per_id.start_frame;
        //     Eigen::Matrix3d R0(Qs[frame_i] * qbc);
        //     Eigen::Vector3d t0 = Ps[frame_i] + Qs[frame_i] * tbc;

        //     // transform points into the frame_i frame to estimate the depth    
        //     frame_i++;
        //     Eigen::Matrix3d R1(Qs[frame_i] * qbc);
        //     Eigen::Vector3d t1 = Ps[frame_i] + Qs[frame_i] * tbc;
        //     Eigen::Matrix3d R = R0.transpose() * R1;
        //     Eigen::Vector3d t = 

        //     Eigen::Vector2d point0, point1;
        //     Eigen::Vector3d point3d;
        //     point0 = it_per_id.feature_per_frame[0].point.head(2);
        //     point1 = it_per_id.feature_per_frame[1].point.head(2);
        //     triangulatePoint(leftPose, rightPose, point0, point1, point3d);
        //     Eigen::Vector3d localPoint;
        //     localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>();
        //     double depth = localPoint.z();
        //     if (depth > 0)
        //     {
        //         it_per_id.estimated_depth = depth;
        //         succ_tagl_cnt++;
        //     }
        //     else
        //     {
        //         it_per_id.estimated_depth = INIT_DEPTH;
        //     }
        //     /*
        //     Vector3d ptsGt = pts_gt[it_per_id.feature_id];
        //     printf("motion  %d pts: %f %f %f gt: %f %f %f \n",it_per_id.feature_id, point3d.x(), point3d.y(), point3d.z(),
        //                                                     ptsGt.x(), ptsGt.y(), ptsGt.z());
        //     */
        //     continue;
        // } 
        // else 
        // {
        // }

        // version of VINS-MONO
//         int frame_i = it_per_id.start_frame, frame_j = frame_i - 1;
//         Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
//         int svd_idx = 0;

//         Eigen::Matrix3d R0(Qs[frame_i] * qbc);
//         Eigen::Vector3d t0 = Ps[frame_i] + Qs[frame_i] * tbc;
//         for (auto &it_per_frame : it_per_id.feature_per_frame)
//         {
//             frame_j++;
//             Eigen::Matrix3d R1(Qs[frame_j] * qbc);
//             Eigen::Vector3d t1 = Ps[frame_j] + Qs[frame_j] * tbc;
//             Eigen::Matrix3d R = R0.transpose() * R1; // from 0 to 1
//             Eigen::Vector3d t = R0.transpose() * (t1 - t0);
//             Eigen::Matrix<double, 3, 4> P;
//             P.leftCols<3>() = R.transpose();
//             P.rightCols<1>() = -R.transpose() * t;
//             Eigen::Vector3d f = it_per_frame.point.normalized();
//             svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0); // TODO:
//             svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);
//             if (frame_i == frame_j)
//                 continue;
//         }
//         ROS_ASSERT(svd_idx == svd_A.rows());
//         Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
//         double svd_method = svd_V[2] / svd_V[3];
//         //it_per_id->estimated_depth = -b / A;
//         //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

//         // printf("depth: %lf\n", svd_method);
//         if (svd_method >= 0.01)
//         {
//             it_per_id.estimated_depth = svd_method;
//             succ_tagl_cnt++;
//         } 
//         else
//         {
//             it_per_id.estimated_depth = INIT_DEPTH;
//         } 
//     }
//     // printFeatureInfo();
//     printf("New triangulation point: %d\n", succ_tagl_cnt);
// }

// http://www.cs.cmu.edu/~16385/s17/Slides/11.4_Triangulation.pdf
// OpenCV function: https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gad3fc9a0c82b08df034234979960b778c
void FeatureManager::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                                      Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d)
{
    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
    Eigen::Vector4d triangulated_point;
    triangulated_point =
        design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

bool FeatureManager::solvePoseByPnP(Eigen::Matrix3d &R, Eigen::Vector3d &P, 
                                    vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D)
{
    // w_T_cam ---> cam_T_w
    Eigen::Matrix3d R_initial;
    Eigen::Vector3d P_initial;
    R_initial = R.inverse();
    P_initial = -(R_initial * P);

    //printf("pnp size %d \n",(int)pts2D.size() );
    if (int(pts2D.size()) < 4)
    {
        printf("feature tracking not enough, please slowly move you device! \n");
        return false;
    }
    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(R_initial, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_initial, t);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    bool pnp_succ = cv::solvePnP(pts3D, pts2D, K, D, rvec, t, 1);
    //pnp_succ = solvePnPRansac(pts3D, pts2D, K, D, rvec, t, true, 100, 8.0 / focalLength, 0.99, inliers);

    if (!pnp_succ)
    {
        printf("pnp failed ! \n");
        return false;
    }

    // bool pnp_succ;
    // // PnP 1:
    // cv::Mat inliers;
    // // pnp_succ = cv::solvePnP(pts3D, pts2D, K, D, rvec, t, true, cv::SOLVEPNP_ITERATIVE);
    // // pnp_succ = cv::solvePnP(pts3D, pts2D, K, D, rvec, t, false, cv::SOLVEPNP_EPNP);
    // pnp_succ = cv::solvePnPRansac(pts3D, pts2D, K, D, rvec, t, true, 100, 8.0 / FOCAL_LENGTH, 0.99, inliers);
    // // std::cout << rvec << std::endl << t << std::endl << std::endl;
    // // std::cout << "number of inliers: " << inliers.size() << std::endl;
    // if (!pnp_succ)
    // {
    //     printf("pnp failed ! \n");
    //     return false;
    // }

    // // PnP 2: only PnP with inliers
    // std::vector<cv::Point2f> pts2D_inliers;
    // std::vector<cv::Point3f> pts3D_inliers;
    // for (size_t i = 0; i < inliers.rows; i++)
    // {
    //     pts2D_inliers.push_back(pts2D[inliers.at<int>(i, 0)]);
    //     pts3D_inliers.push_back(pts3D[inliers.at<int>(i, 0)]);
    // }
    // pnp_succ = cv::solvePnPRansac(pts3D_inliers, pts2D_inliers, K, D, rvec, t, 1);
    // if (!pnp_succ)
    // {
    //     printf("pnp failed ! \n");
    //     return false;
    // }

    cv::Rodrigues(rvec, r);
    //cout << "r " << endl << r << endl;
    Eigen::MatrixXd R_pnp;
    cv::cv2eigen(r, R_pnp);
    Eigen::MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);

    // cam_T_w ---> w_T_cam
    R = R_pnp.transpose();
    P = -(R * T_pnp);
    return true;
}

// applied to stereo, directly triangulate
void FeatureManager::initFramePoseByPnP(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[])
{
    if (frameCnt > 0)
    {
        vector<cv::Point2f> pts2D;
        vector<cv::Point3f> pts3D;
        for (auto &it_per_id : feature_)
        {
            if (it_per_id.estimated_depth > 0)
            {
                int index = frameCnt - it_per_id.start_frame;
                if ((int)it_per_id.feature_per_frame.size() >= index + 1)
                {
                    // predict
                    Vector3d ptsInCam = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
                    Vector3d ptsInBody = ric[0] * ptsInCam + tic[0];
                    Vector3d ptsInWorld = Rs[it_per_id.start_frame] * ptsInBody + Ps[it_per_id.start_frame];
                    cv::Point3f point3d(ptsInWorld.x(), ptsInWorld.y(), ptsInWorld.z());
                    pts3D.push_back(point3d);
                    // measurement
                    cv::Point2f point2d(it_per_id.feature_per_frame[index].point.x(),
                                        it_per_id.feature_per_frame[index].point.y());
                    pts2D.push_back(point2d);
                }
            }
        }
        // printf("%lu features are used for PnP\n", pts2D.size());

        // trans to w_T_cam
        Eigen::Matrix3d RCam;
        Eigen::Vector3d PCam;
        RCam = Rs[frameCnt - 1] * ric[0];
        PCam = Rs[frameCnt - 1] * tic[0] + Ps[frameCnt - 1];
        if (solvePoseByPnP(RCam, PCam, pts2D, pts3D))
        {
            // trans to w_T_imu
            Rs[frameCnt] = RCam * ric[0].transpose();
            Ps[frameCnt] = -RCam * ric[0].transpose() * tic[0] + PCam;
            // Eigen::Quaterniond Q(Rs[frameCnt]);
            //cout << "frameCnt: " << frameCnt <<  " pnp Q " << Q.w() << " " << Q.vec().transpose() << endl;
            //cout << "frameCnt: " << frameCnt << " pnp P " << Ps[frameCnt].transpose() << endl;
        }
    }
}

void FeatureManager::removeBackShiftDepth(const Eigen::Quaterniond &marg_Q, const Eigen::Vector3d &marg_P,
                                          const Eigen::Quaterniond &new_Q, const Eigen::Vector3d &new_P)
{
    for (auto it = feature_.begin(), it_next = feature_.begin();
         it != feature_.end(); it = it_next)
    {
        it_next++;
        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() < 2)
            {
                feature_.erase(it);
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_Q * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_Q.inverse() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature_.erase(it);
        }
        */
    }
}

void FeatureManager::savePNPData(const int &frameCnt,
                                 const CircularBuffer<Eigen::Vector3d> &Ps, const CircularBuffer<Eigen::Quaterniond> &Qs,
                                 const Eigen::Vector3d &tbc, const Eigen::Quaterniond &qbc, const cv::Mat &img)
{
    if (frameCnt > 0)
    {
        vector<cv::Point2f> pts2D;
        vector<cv::Point3f> pts3D;
        vector<cv::Point3f> pts3D_proj;
        vector<cv::Point2f> pts_uv;
        for (auto &it_per_id : feature_)
        {
            if (it_per_id.estimated_depth > 0)
            {
                int index = frameCnt - it_per_id.start_frame;
                if ((int)it_per_id.feature_per_frame.size() >= index + 1)
                {
                    Vector3d ptsInCam = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
                    Vector3d ptsInBody = qbc * ptsInCam + tbc;
                    Vector3d ptsInWorld = Qs[it_per_id.start_frame] * ptsInBody + Ps[it_per_id.start_frame];

                    cv::Point3f point3d(ptsInWorld.x(), ptsInWorld.y(), ptsInWorld.z());
                    pts3D.push_back(point3d);
                    cv::Point2f point2d(it_per_id.feature_per_frame[index].point.x(),
                                        it_per_id.feature_per_frame[index].point.y());
                    pts2D.push_back(point2d);
                    cv::Point2f point_uv(it_per_id.feature_per_frame[index].uv[0],
                                         it_per_id.feature_per_frame[index].uv[1]);
                    pts_uv.push_back(point_uv);
                }
            }
        }

        std::ofstream ofs;
        ofs.open("/home/jjiao/Documents/matlab_ws/eloam/data/pnp_data/pts3d.txt", std::ios::out | std::ios::binary);
        if (!ofs.is_open())
        {
            exit(EXIT_FAILURE);
        }
        for (size_t i = 0; i < pts3D.size(); i++)
        {
            ofs << pts3D[i].x << " " << pts3D[i].y << " " << pts3D[i].z << std::endl;
        }
        ofs.close();

        ofs.open("/home/jjiao/Documents/matlab_ws/eloam/data/pnp_data/pts2d.txt", std::ios::out | std::ios::binary);
        if (!ofs.is_open())
        {
            exit(EXIT_FAILURE);
        }
        for (size_t i = 0; i < pts2D.size(); i++)
        {
            ofs << pts2D[i].x << " " << pts2D[i].y << std::endl;
        }
        ofs.close();

        ofs.open("/home/jjiao/Documents/matlab_ws/eloam/data/pnp_data/ptsuv.txt", std::ios::out | std::ios::binary);
        if (!ofs.is_open())
        {
            exit(EXIT_FAILURE);
        }
        for (size_t i = 0; i < pts_uv.size(); i++)
        {
            ofs << pts_uv[i].x << " " << pts_uv[i].y << std::endl;
        }
        ofs.close();

        cv::imwrite("/home/jjiao/Documents/matlab_ws/eloam/data/pnp_data/cur_frame.png", img);

        ofs.open("/home/jjiao/Documents/matlab_ws/eloam/data/pnp_data/pose_est.txt", std::ios::out | std::ios::binary);
        Eigen::Quaterniond q_w_T_cam = Qs[frameCnt] * qbc;
        Eigen::Vector3d t_w_T_cam = Qs[frameCnt] * tbc + Ps[frameCnt];
        if (!ofs.is_open())
        {
            exit(EXIT_FAILURE);
        }
        ofs << "tx, ty, tz, qw, qx, qy, qz" << std::endl;
        ofs << t_w_T_cam[0] << " " << t_w_T_cam[1] << " " << t_w_T_cam[2] << " " << q_w_T_cam.w() << " " << q_w_T_cam.x() << " " << q_w_T_cam.y() << " " << q_w_T_cam.z() << std::endl;
        ofs.close();

        ofs.open("/home/jjiao/Documents/matlab_ws/eloam/data/pnp_data/pts3d_proj.txt", std::ios::out | std::ios::binary);
        if (!ofs.is_open())
        {
            exit(EXIT_FAILURE);
        }
        for (size_t i = 0; i < pts3D.size(); i++)
        {
            Eigen::Vector3d pt(pts3D[i].x, pts3D[i].y, pts3D[i].z);
            pt = q_w_T_cam.inverse() * (pt - t_w_T_cam); // project to the normalized camera plane
            ofs << pt[0] / pt[2] << " " << pt[1] / pt[2] << std::endl;
        }
        ofs.close();

        // compute the reprojection error after PnP
        double reproj_err = 0;
        for (size_t i = 0; i < pts3D.size(); i++)
        {
            Eigen::Vector3d pts_proj(pts3D[i].x, pts3D[i].y, pts3D[i].z);
            pts_proj = q_w_T_cam.inverse() * (pts_proj - t_w_T_cam); // project to the normalized camera plane
            pts_proj /= pts_proj[2];
            Eigen::Vector2d pts(pts2D[i].x, pts2D[i].y);
            reproj_err += (pts - pts_proj.head<2>()).norm();
        }
        printf("average reprojection error after PnP: %lf\n", reproj_err / pts3D.size());
    }
}

// find correspondences between frame_count_l <-> frame_count_
vector<pair<Vector2d, Vector2d>> FeatureManager::getCorresponding2D(const int &frame_count_l, const int &frame_count_r)
{
    vector<pair<Vector2d, Vector2d>> corres;
    for (auto &it : feature_)
    {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector2d a = Vector2d::Zero(), b = Vector2d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;
            a = it.feature_per_frame[idx_l].uv;
            b = it.feature_per_frame[idx_r].uv;
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::printFeatureInfo() const
{
    size_t cnt = 0;
    for (const auto &it: feature_)
    {
        printf("%luth: start_frame: %d, observations: %lu\n", cnt, it.start_frame, it.feature_per_frame.size());
        cnt++;        
    }
}