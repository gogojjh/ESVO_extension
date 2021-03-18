#include <glog/logging.h>

#include <ros/ros.h>
#include <std_msgs/Time.h>

#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>

#include <deque>
#include <mutex>
#include <unistd.h>

int newEvents;

template <typename T>
T param(const ros::NodeHandle &nh, const std::string &name, const T &defaultValue)
{
    if (nh.hasParam(name))
    {
        T v;
        nh.param<T>(name, v, defaultValue);
        ROS_INFO_STREAM("Found parameter: " << name << ", value: " << v);
        return v;
    }
    ROS_WARN_STREAM("Cannot find value for parameter: " << name << ", assigning default: " << defaultValue);
    return defaultValue;
}

void eventsCallback(const dvs_msgs::EventArray::ConstPtr &msg)
{
    newEvents += msg->events.size();
}

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "esvo_time_surface_global_timer");

    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    const int MINIMUM_EVENTS = param(nh_private, "minimum_events", static_cast<int>(1000));
    const int FREQUENCY_TIMER = param(nh_private, "frequency_timer", static_cast<int>(100));
    const int MINIMUM_FREQUENCY_TIMER = param(nh_private, "minimum_frequency_timer", static_cast<int>(40));
    ros::Subscriber event_sub = nh.subscribe("events", 0, eventsCallback);
    ros::Publisher global_time_pub = nh.advertise<std_msgs::Time>("/sync", 1);

    newEvents = 0;
    double timestampLast = ros::WallTime::now().toSec();
    while (ros::ok())
    {
        ros::spinOnce();
        if ((ros::WallTime::now().toSec() - timestampLast >= 1.0 / FREQUENCY_TIMER) && (newEvents >= MINIMUM_EVENTS)
         || (ros::WallTime::now().toSec() - timestampLast >= 1.0 / MINIMUM_FREQUENCY_TIMER))
        {
            // LOG_EVERY_N(INFO, 10) << ros::WallTime::now().toSec() - timestampLast;
            std_msgs::Time msg;
            msg.data = ros::Time(ros::WallTime::now().toSec());
            global_time_pub.publish(msg);
            timestampLast = msg.data.toSec();
            newEvents = 0;
        }
        usleep(1e3); // 1ms
    }
    return 0;
}
