#include <initial/MonoInitializer.h>

using namespace std;
using namespace Eigen;

const int MAX_CNT = 200;
const bool SHOW_TRACK = true;
const double MIN_DIST = 15;
const bool FLOW_BACK = true;
const size_t MIN_PARALLAX = 5;

double ptsDistance(const cv::Point2f &pt1, const cv::Point2f &pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

bool MonoInitializer::inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return (BORDER_SIZE <= img_x && img_x < col_ - BORDER_SIZE &&
            BORDER_SIZE <= img_y && img_y < row_ - BORDER_SIZE);
}

void MonoInitializer::setMask()
{
    mask_ = cv::Mat(row_, col_, CV_8UC1, cv::Scalar(255));

    // prefer to keep features that are tracked for long time
    std::vector<std::pair<int, std::pair<cv::Point2f, int>>> cnt_pts_id;
    for (size_t i = 0; i < cur_pts_.size(); i++)
        cnt_pts_id.push_back(std::make_pair(track_cnt_[i], std::make_pair(cur_pts_[i], ids_[i])));

    std::sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const std::pair<int, std::pair<cv::Point2f, int>> &a, const std::pair<int, std::pair<cv::Point2f, int>> &b) {
        return a.first > b.first;
    });

    cur_pts_.clear();
    ids_.clear();
    track_cnt_.clear();
    for (auto &it : cnt_pts_id)
    {
        if (mask_.at<uchar>(it.second.first) == 255)
        {
            cur_pts_.push_back(it.second.first);
            ids_.push_back(it.second.second);
            track_cnt_.push_back(it.first);
            cv::circle(mask_, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void MonoInitializer::drawTrack()
{
    img_track_vis_ = cur_img_.clone();
    cv::cvtColor(img_track_vis_, img_track_vis_, CV_GRAY2RGB);
    for (size_t j = 0; j < cur_pts_.size(); j++)
    {
        double len = std::min(1.0, 1.0 * track_cnt_[j] / 20);
        cv::circle(img_track_vis_, cur_pts_[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }
    for (size_t i = 0; i < prev_pts_.size(); i++)
    {
        cv::arrowedLine(img_track_vis_, prev_pts_[i], cur_pts_[i], cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
    }
    // std::cout << "drawTrack" << std::endl;
}

MonoInitializer::MonoInitializer()
{
    n_id_ = 0;
    has_prediction_ = false;
    img_track_vis_ = cv::Mat::zeros(1, 1, CV_8UC3);
    prev_pts_.clear();
    cur_pts_.clear();
    prev_un_pts_.clear();
    cur_un_pts_.clear();
    track_cnt_.clear();
    ids_.clear();
    prev_time_ = 0;
    cur_time_ = 0;

    // f_manager_.clearState();
}

void MonoInitializer::trackImage(const double &t, const cv::Mat &img, const Eigen::Matrix3d &K)
{
    cur_time_ = t;
    row_ = img.rows;
    col_ = img.cols;
    cur_img_ = img.clone();
    cur_pts_.clear();

    if (prev_pts_.size() > 0)
    {
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(prev_img_, cur_img_, prev_pts_, cur_pts_, status, err, cv::Size(21, 21), 3);

        // reverse check, double check
        if (FLOW_BACK)
        {
            std::vector<uchar> reverse_status;
            std::vector<cv::Point2f> reverse_pts = prev_pts_;
            cv::calcOpticalFlowPyrLK(cur_img_, prev_img_, cur_pts_, reverse_pts, reverse_status, err, cv::Size(21, 21), 1,
                                     cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
            for (size_t i = 0; i < status.size(); i++)
            {
                if (status[i] && reverse_status[i] && ptsDistance(prev_pts_[i], reverse_pts[i]) <= 0.5)
                {
                    status[i] = 1;
                }
                else
                {
                    status[i] = 0;
                }
            }
        }
        for (size_t i = 0; i < cur_pts_.size(); i++)
            if (status[i] && !inBorder(cur_pts_[i]))
                status[i] = 0;
        reduceVector(prev_pts_, status);
        reduceVector(cur_pts_, status);
        reduceVector(ids_, status);
        reduceVector(track_cnt_, status);
        if (SHOW_TRACK)
            drawTrack();
    }
    for (auto &n : track_cnt_)
        n++;

    if (1)
    {
        setMask();
        int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts_.size());
        if (n_max_cnt > 0)
        {
            if (mask_.empty())
                std::cout << "mask is empty " << std::endl;
            if (mask_.type() != CV_8UC1)
                std::cout << "mask type wrong " << std::endl;
            cv::goodFeaturesToTrack(cur_img_, n_pts_, n_max_cnt, 0.01, MIN_DIST, mask_);
        }
        else
            n_pts_.clear();

        // add new points
        for (auto &p : n_pts_)
        {
            cur_pts_.push_back(p);
            ids_.push_back(n_id_++);
            track_cnt_.push_back(1);
        }
        // LOG(INFO) << "detect corner pts size: " << cur_pts_.size();
    }

    // project the pixel into the projective plane (no distorted points)
    cur_un_pts_.clear(); 
    std::vector<cv::Point2f> un_pts;
    for (size_t i = 0; i < cur_pts_.size(); i++)
    {
        Eigen::Vector2d p(cur_pts_[i].x, cur_pts_[i].y);
        // Lift points to normalised plane
        double mx_u = K(0, 0) * p(0) + K(0, 2);
        double my_u = K(1, 1) * p(1) + K(1, 2);
        Eigen::Vector3d P(mx_u, my_u, 1.0);
        cur_un_pts_.push_back(cv::Point2f(P.x() / P.z(), P.y() / P.z()));
    }
    focal_length_ = K(0, 0);

    prev_img_ = cur_img_.clone();
    prev_pts_ = cur_pts_;
    prev_un_pts_ = cur_un_pts_;
    prev_time_ = cur_time_;
    has_prediction_ = false;

    // combined feature vector: [x, y, z, p_u, p_v]
    feature_vec_.clear();
    for (size_t i = 0; i < ids_.size(); i++)
    {
        int feature_id = ids_[i];
        int camera_id = 0;
        double x, y, z;
        x = cur_un_pts_[i].x;
        y = cur_un_pts_[i].y;
        z = 1.0;
        double p_u, p_v;
        p_u = cur_pts_[i].x;
        p_v = cur_pts_[i].y;

        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv << x, y, z, p_u, p_v, 0.0, 0.0;
        feature_vec_[feature_id].emplace_back(camera_id, xyz_uv_velocity);
    }
}

// void MonoInitializer::addNewFrame(const double &t)
// {
    // if (f_manager_.addFeatureCheckParallax(frame_count_, feature_vec_))
    //     isLargeParallax_ = true;    
    // else
    //     isLargeParallax_ = false;
    // Headers[frame_count_] = t;
    // allKeyFrame[frame_count_] = feature_vec_;
// }

// void MonoInitializer::solveRelativePose()
// {
    // Eigen::Matrix3d relative_R;
    // Eigen::Vector3d relative_T;
    // int l;
    // bool solveFlag = false;
    // for (int i = 0; i < WINDOW_SIZE; i++)
    // {
    //     vector<pair<Vector3d, Vector3d>> corres;
    //     corres = f_manager_.getCorresponding(i, WINDOW_SIZE);
    //     if (corres.size() > 20)
    //     {
    //         double sum_parallax = 0;
    //         double average_parallax;
    //         for (int j = 0; j < int(corres.size()); j++)
    //         {
    //             Eigen::Vector2d pts_0(corres[j].first(0), corres[j].first(1));
    //             Eigen::Vector2d pts_1(corres[j].second(0), corres[j].second(1));
    //             double parallax = (pts_0 - pts_1).norm();
    //             sum_parallax = sum_parallax + parallax;
    //         }
    //         average_parallax = 1.0 * sum_parallax / int(corres.size());
    //         if (average_parallax * 460 > 30 && m_estimator_.solveRelativeRT(corres, relative_R, relative_T))
    //         {
    //             l = i;
    //             LOG(INFO) << "average_parallax " << average_parallax * 460 << "choose " << l << " and newest frame to triangulate the whole structure";
    //             solveFlag = true;
    //         }
    //     }
    // }

    // if (solveFlag)
    // {

    // }
// }

bool MonoInitializer::initialize(const Eigen::Matrix3d &K)
{
    // cv::Mat K_mat;
    // cv::eigen2cv(K, K_mat);
    // cv::Mat motion;
    // if (modelSelector_.select(K_mat, prev_pts_, cur_pts_, motion))
    // {
    //     Eigen::Matrix4d eigenPose;
    //     eigenPose(0, 0) = motion.at<float>(0, 0);
    //     eigenPose(0, 1) = motion.at<float>(0, 1);
    //     eigenPose(0, 2) = motion.at<float>(0, 2);
    //     eigenPose(0, 3) = motion.at<float>(0, 3);
    //     eigenPose(1, 0) = motion.at<float>(1, 0);
    //     eigenPose(1, 1) = motion.at<float>(1, 1);
    //     eigenPose(1, 2) = motion.at<float>(1, 2);
    //     eigenPose(1, 3) = motion.at<float>(1, 3);
    //     eigenPose(2, 0) = motion.at<float>(2, 0);
    //     eigenPose(2, 1) = motion.at<float>(2, 1);
    //     eigenPose(2, 2) = motion.at<float>(2, 2);
    //     eigenPose(2, 3) = motion.at<float>(2, 3);
    //     eigenPose(3, 0) = motion.at<float>(3, 0);
    //     eigenPose(3, 1) = motion.at<float>(3, 1);
    //     eigenPose(3, 2) = motion.at<float>(3, 2);
    //     eigenPose(3, 3) = motion.at<float>(3, 3);

    //     std::cout << eigenPose << std::endl;
    //     std::cout << "parallax: " << modelSelector_.parallaxBest << std::endl;
    // }

    // prev_img_ = cur_img_.clone();
    // prev_pts_ = cur_pts_;
    // prev_time_ = cur_time_;

    // global sfm
    // add features for the global sfm
    // map<int, Vector3d> sfm_tracked_points;
    // vector<SFMFeature> sfm_f;
    // for (auto &it_per_id : f_manager.feature_)
    // {
    //     int frame_j = it_per_id.start_frame - 1;
    //     SFMFeature tmp_feature;
    //     tmp_feature.state = false;
    //     tmp_feature.id = it_per_id.feature_id;
    //     for (auto &it_per_frame : it_per_id.feature_per_frame)
    //     {
    //         frame_j++;
    //         Vector3d pts_j = it_per_frame.point;
    //         tmp_feature.observation.push_back(make_pair(frame_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
    //     }
    //     sfm_f.push_back(tmp_feature);
    // }

    // Matrix3d relative_R;
    // Vector3d relative_T;
    // int l;
    // if (!relativePose(relative_R, relative_T, l))
    // {
    //     ROS_INFO("Not enough features or parallax; Move device around");
    //     return false;
    // }

    // Quaterniond Q[frame_count + 1];
    // Vector3d T[frame_count + 1];
    // GlobalSFM sfm;
    // double depth_min, depth_median;
    // if (!sfm.construct(frame_count + 1, Q, T, l, relative_R, relative_T,
    //                    sfm_f, sfm_tracked_points, depth_min, depth_median))
    // {
    //     ROS_DEBUG("global SFM failed!");
    //     marginalization_flag = MARGIN_OLDEST_KF;
    //     return false;
    // }

    // // recover the states for all frame
    // map<double, ImageFrame>::iterator frame_it;
    // map<int, Vector3d>::iterator it;
    // frame_it = all_image_frame.begin();
    // for (int i = 0; frame_it != all_image_frame.end(); frame_it++)
    // {
    //     // provide initial guess of keyframes
    //     cv::Mat r, rvec, t, D, tmp_r;
    //     if ((frame_it->first) == Headers[i].stamp.toSec())
    //     {
    //         frame_it->second.is_key_frame = true;
    //         frame_it->second.R = Q[i].toRotationMatrix();
    //         frame_it->second.T = T[i];
    //         i++;
    //         continue;
    //     }

    //     // provide initial guess of non-keyframes using PnP
    //     if ((frame_it->first) > Headers[i].stamp.toSec())
    //     {
    //         i++;
    //     }
    //     Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
    //     Vector3d P_inital = -R_inital * T[i];
    //     cv::eigen2cv(R_inital, tmp_r);
    //     cv::Rodrigues(tmp_r, rvec);
    //     cv::eigen2cv(P_inital, t);

    //     frame_it->second.is_key_frame = false;
    //     vector<cv::Point3f> pts_3_vector;
    //     vector<cv::Point2f> pts_2_vector;
    //     for (auto &id_pts : frame_it->second.points)
    //     {
    //         int feature_id = id_pts.first;
    //         for (auto &i_p : id_pts.second)
    //         {
    //             it = sfm_tracked_points.find(feature_id);
    //             if (it != sfm_tracked_points.end())
    //             {
    //                 Vector3d world_pts = it->second;
    //                 cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
    //                 pts_3_vector.push_back(pts_3);
    //                 Vector2d img_pts = i_p.second.head<2>();
    //                 cv::Point2f pts_2(img_pts(0), img_pts(1));
    //                 pts_2_vector.push_back(pts_2);
    //             }
    //         }
    //     }
    //     cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    //     if (pts_3_vector.size() < 6)
    //     {
    //         cout << "pts_3_vector size " << pts_3_vector.size() << endl;
    //         ROS_DEBUG("Not enough points for solve pnp !");
    //         return false;
    //     }
    //     if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
    //     {
    //         ROS_DEBUG("solve pnp fail!");
    //         return false;
    //     }
    //     cv::Rodrigues(rvec, r);
    //     MatrixXd R_pnp, tmp_R_pnp;
    //     cv::cv2eigen(r, tmp_R_pnp);
    //     R_pnp = tmp_R_pnp.transpose();
    //     MatrixXd T_pnp;
    //     cv::cv2eigen(t, T_pnp);
    //     T_pnp = R_pnp * (-T_pnp);
    //     frame_it->second.R = R_pnp;
    //     frame_it->second.T = T_pnp;
    // }

    // // align the motion and acquire the absolute scale
    // for (int i = 0; i <= frame_count; i++)
    // {
    //     Matrix3d Ri = all_image_frame[Headers[i].stamp.toSec()].R;
    //     Vector3d Pi = all_image_frame[Headers[i].stamp.toSec()].T;
    //     Rs[i] = Ri;
    //     Ps[i] = Pi;
    //     all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
    // }

    // for (size_t i = 0; i <= frame_count; i++)
    // {
    //     Matrix3d R_cam_i_T_cam_j = Rs[0].transpose() * Rs[i];
    //     Vector3d t_cam_i_T_cam_j = Rs[0].transpose() * (s * Ps[i] - s * Ps[0]);
    //     Rs[i] = ric[0] * R_cam_i_T_cam_j * ric[0].transpose();
    //     Ps[i] = -ric[0] * R_cam_i_T_cam_j * ric[0].transpose() * tic[0] + ric[0] * t_cam_i_T_cam_j + tic[0];
    // }
    // return true;
}

// find the relative R, t between the earliest frame which contians enough correspondance and parallex with the newest frame
// bool MonoInitializer::relativePose(Eigen::Matrix3d &relative_R, Eigen::Vector3d &relative_T, int &l)
// {
//     for (int i = 0; i < WINDOW_SIZE; i++)
//     {
//         vector<pair<Vector3d, Vector3d>> corres;
//         corres = f_manager.getCorresponding(i, WINDOW_SIZE);
//         if (corres.size() > 20)
//         {
//             double sum_parallax = 0;
//             double average_parallax;
//             for (int j = 0; j < int(corres.size()); j++)
//             {
//                 Eigen::Vector2d pts_0(corres[j].first(0), corres[j].first(1));
//                 Eigen::Vector2d pts_1(corres[j].second(0), corres[j].second(1));
//                 double parallax = (pts_0 - pts_1).norm();
//                 sum_parallax = sum_parallax + parallax;
//             }
//             average_parallax = 1.0 * sum_parallax / int(corres.size());
//             if (average_parallax * 460 > 30 && m_estimator_.solveRelativeRT(corres, relative_R, relative_T))
//             {
//                 l = i;
//                 LOG(INFO) << "average_parallax " << average_parallax * 460 << "choose " << l << " and newest frame to triangulate the whole structure";
//                 return true;
//             }
//         }
//     }
//     return false;
// }