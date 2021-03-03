#include <initial/MonoInitializer.h>

const int MAX_CNT = 200;
const bool SHOW_TRACK = true;
const double MIN_DIST = 15;

void reduceVector(std::vector<cv::Point2f> &v, std::vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

bool MonoInitializer::inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return (BORDER_SIZE <= img_x && img_x < col_ - BORDER_SIZE &&
            BORDER_SIZE <= img_y && img_y < row_ - BORDER_SIZE);
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
    feature_vec_.clear();
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
        LOG(INFO) << "detect corner pts size: " << cur_pts_.size();
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

    // set prev frame
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

        Eigen::Matrix<double, 5, 1> xyz_uv;
        xyz_uv << x, y, z, p_u, p_v;
        feature_vec_[feature_id].emplace_back(camera_id, xyz_uv);
    }
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