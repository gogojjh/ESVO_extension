#include <iostream>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>

size_t N_accumEvents;
size_t accumulateEvent;
pcl::PointCloud<pcl::PointXYZ>::Ptr PC_ptr;

std::string resultPath;
std::string strDataset;
std::string strSequence;

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
    accumulateEvent += msg->events.size();
}

void refMapCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    pcl::fromROSMsg(*msg, *PC_ptr);
    if (accumulateEvent > N_accumEvents)
    {
        std::string pcdName = resultPath + strDataset + "/" + strSequence + ".pcd";
        pcl::io::savePCDFileASCII(pcdName, *PC_ptr);
        std::cout << "Saving point cloud: " << PC_ptr->size() << std::endl;
        ros::shutdown();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "saveMap");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    N_accumEvents = param(nh_private, "accumulate_events", 4e5);
    resultPath = param(nh_private, "PATH_TO_SAVE_TRAJECTORY", std::string("~/ESVO_result"));
    strDataset = param(nh_private, "DATASET_NAME", std::string("rpg"));
    strSequence = param(nh_private, "SEQUENCE_NAME", std::string("shapes_poster"));

    ros::Subscriber events_left_sub = nh.subscribe<dvs_msgs::EventArray>("events_left", 0, eventsCallback);
    ros::Subscriber map_sub = nh.subscribe("pointcloud", 0, refMapCallback);
    PC_ptr.reset(new pcl::PointCloud<pcl::PointXYZ>());
    accumulateEvent = 0;

    ros::Rate r(100);
    while (ros::ok())
    {
        ros::spinOnce();
        r.sleep();
    }
    return 0;
}
