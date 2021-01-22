// ****************************************************************
// ***************** Event Processing
// ****************************************************************
/**
 * @brief: Process incoming events
 */
void TimeSurface::eventsCallback(const dvs_msgs::EventArray::ConstPtr &msg)
{
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (!bSensorInitialized_)
        init(msg->width, msg->height);

    for (const dvs_msgs::Event &e : msg->events)
    {
        events_.push_back(e);
        int i = events_.size() - 2;
        while (i >= 0 && events_[i].ts > e.ts)
        {
            events_[i + 1] = events_[i];
            i--;
        }
        events_[i + 1] = e;

        const dvs_msgs::Event &last_event = events_.back();
        pEventQueueMat_->insertEvent(last_event);
    }
    clearEventQueue();
}

void TimeSurface::clearEventQueue()
{
    static constexpr size_t MAX_EVENT_QUEUE_LENGTH = 5000000;
    if (events_.size() > MAX_EVENT_QUEUE_LENGTH)
    {
        size_t remove_events = events_.size() - MAX_EVENT_QUEUE_LENGTH;
        events_.erase(events_.begin(), events_.begin() + remove_events);
    }
}

/**
 * @brief: generate Time-Surface Maps
 * equ. (1)
 */
void TimeSurface::createTimeSurfaceAtTime(const ros::Time &external_sync_time)
{
    // ...

    // create exponential-decayed Time Surface map.
    const double decay_sec = decay_ms_ / 1000.0;
    cv::Mat time_surface_map;
    time_surface_map = cv::Mat::zeros(sensor_size_, CV_64F);

    // Loop through all coordinates
    for (int y = 0; y < sensor_size_.height; ++y)
    {
        for (int x = 0; x < sensor_size_.width; ++x)
        {
            dvs_msgs::Event most_recent_event_at_coordXY_before_T;
            if (pEventQueueMat_->getMostRecentEventBeforeT(x, y, external_sync_time, &most_recent_event_at_coordXY_before_T))
            {
                const ros::Time &most_recent_stamp_at_coordXY = most_recent_event_at_coordXY_before_T.ts;
                if (most_recent_stamp_at_coordXY.toSec() > 0) // check if negative timestamp
                {
                    const double dt = (external_sync_time - most_recent_stamp_at_coordXY).toSec();
                    double polarity = (most_recent_event_at_coordXY_before_T.polarity) ? 1.0 : -1.0;
                    double expVal = std::exp(-dt / decay_sec);
                    if (!ignore_polarity_)
                        expVal *= polarity;

                    // Backward version
                    if (time_surface_mode_ == BACKWARD)
                        time_surface_map.at<double>(y, x) = expVal;

                    // Forward version
                    if (time_surface_mode_ == FORWARD && bCamInfoAvailable_)
                    {
                        Eigen::Matrix<double, 2, 1> uv_rect = precomputed_rectified_points_.block<2, 1>(0, y * sensor_size_.width + x);
                        size_t u_i, v_i;
                        if (uv_rect(0) >= 0 && uv_rect(1) >= 0)
                        {
                            u_i = std::floor(uv_rect(0));
                            v_i = std::floor(uv_rect(1));

                            if (u_i + 1 < sensor_size_.width && v_i + 1 < sensor_size_.height)
                            {
                                double fu = uv_rect(0) - u_i;
                                double fv = uv_rect(1) - v_i;
                                double fu1 = 1.0 - fu;
                                double fv1 = 1.0 - fv;
                                time_surface_map.at<double>(v_i, u_i) += fu1 * fv1 * expVal;
                                time_surface_map.at<double>(v_i, u_i + 1) += fu * fv1 * expVal;
                                time_surface_map.at<double>(v_i + 1, u_i) += fu1 * fv * expVal;
                                time_surface_map.at<double>(v_i + 1, u_i + 1) += fu * fv * expVal;

                                if (time_surface_map.at<double>(v_i, u_i) > 1)
                                    time_surface_map.at<double>(v_i, u_i) = 1;
                                if (time_surface_map.at<double>(v_i, u_i + 1) > 1)
                                    time_surface_map.at<double>(v_i, u_i + 1) = 1;
                                if (time_surface_map.at<double>(v_i + 1, u_i) > 1)
                                    time_surface_map.at<double>(v_i + 1, u_i) = 1;
                                if (time_surface_map.at<double>(v_i + 1, u_i + 1) > 1)
                                    time_surface_map.at<double>(v_i + 1, u_i + 1) = 1;
                            }
                        }
                    } // forward
                }
            } // a most recent event is available
        }     // loop x
    }         // loop y
    // ...
}

