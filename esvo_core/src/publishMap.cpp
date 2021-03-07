#include <iostream>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <dvs_msgs/Event.h>

size_t N_accumEvents;
size_t accumulateEvent;
pcl::PointCloud<pcl::PointXYZ>::Ptr PC_ptr;

std::string resultPath;
std::string strDataset;
std::string strSequence;
size_t mapping_rate_hz;
std::string world_frame_id;

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

int main(int argc, char **argv)
{
    ros::init(argc, argv, "publishMap");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    resultPath = param(nh_private, "PATH_TO_SAVE_TRAJECTORY", std::string("~/ESVO_result"));
    strDataset = param(nh_private, "DATASET_NAME", std::string("rpg"));
    strSequence = param(nh_private, "SEQUENCE_NAME", std::string("shapes_poster"));
    mapping_rate_hz = param(nh_private, "mapping_rate_hz", 20);
    world_frame_id = param(nh_private, "world_frame_id", std::string("world"));

    ros::Publisher pc_pub = nh.advertise<sensor_msgs::PointCloud2>("esvo_MonoMapping/point_cloud_local", 1);

    std::string pcdName = resultPath + "/" + strDataset + "/" + strSequence + ".pcd";
    PC_ptr.reset(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::io::loadPCDFile(pcdName, *PC_ptr);
    PC_ptr->header.frame_id = world_frame_id;
    std::cout << "Loading point cloud: " << PC_ptr->size() << std::endl;

    ros::Rate r(mapping_rate_hz);
    while (ros::ok())
    {
        sensor_msgs::PointCloud2::Ptr pc_to_publish(new sensor_msgs::PointCloud2);
        if (!PC_ptr->empty())
        {
            pcl::toROSMsg(*PC_ptr, *pc_to_publish);
            pc_to_publish->header.stamp = ros::Time::now();
            pc_pub.publish(pc_to_publish);
        }
        r.sleep();
    }
    return 0;
}
