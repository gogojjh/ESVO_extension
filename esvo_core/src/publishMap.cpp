#include <iostream>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>
#include <tf/tf.h>

pcl::PointCloud<pcl::PointXYZ>::Ptr PC_ptr;

std::string resultPath;
std::string strDataset;
std::string strSequence;
size_t mapping_rate_hz;
std::string world_frame_id, dvs_frame_id;
std::string strRep;
size_t TS_start, Event_start;

ros::Time TS_latest_timestamp, EM_latest_timestamp;
size_t TS_number = 0, event_number = 0;
std::deque<sensor_msgs::Image> TS_buf;
std::shared_ptr<tf::Transformer> tf_ptr;

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

void timeSurfaceCallback(const sensor_msgs::ImageConstPtr &time_surface_left)
{
    TS_buf.push_back(*time_surface_left);
    while (TS_buf.size() > 100)
    {
        TS_buf.pop_front();
    }
    TS_number++;
}

void eventsCallback(const dvs_msgs::EventArray::ConstPtr &msg)
{
    EM_latest_timestamp = msg->header.stamp;
    event_number += msg->events.size();
}

bool getPoseAt(const ros::Time &t, const std::string &source_frame)
{
    std::string *err_msg = new std::string();
    if (!tf_ptr->canTransform(world_frame_id, source_frame, t, err_msg))
    {
        // std::cout << t << " : " << *err_msg << std::endl;
        delete err_msg;
        return false;
    }
    else
    {
        return true;
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "publishMap");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    resultPath = param(nh_private, "PATH_TO_SAVE_TRAJECTORY", std::string("/tmp/"));
    strDataset = param(nh_private, "Dataset_Name", std::string("rpg"));
    strSequence = param(nh_private, "Sequence_Name", std::string("shapes_poster"));
    mapping_rate_hz = param(nh_private, "mapping_rate_hz", 20);
    world_frame_id = param(nh_private, "world_frame_id", std::string("map"));
    dvs_frame_id = param(nh_private, "dvs_frame_id", std::string("dvs"));
    strRep = param(nh_private, "Representation_Name", std::string("TS"));
    TS_start = param(nh_private, "TS_start", 20);
    Event_start = param(nh_private, "Event_start", 10000);

    ros::Subscriber TS_left_sub_ = nh.subscribe("time_surface_left", 10, timeSurfaceCallback);
    ros::Subscriber events_left_sub = nh.subscribe("events_left", 10, eventsCallback);
    ros::Publisher pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/publishMap/pointcloud_local", 1);

    tf_ptr = std::make_shared<tf::Transformer>(true, ros::Duration(100.0));

    std::string pcdName = resultPath + "/" + strDataset + "/" + strSequence + ".pcd";
    PC_ptr.reset(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::io::loadPCDFile(pcdName, *PC_ptr);
    PC_ptr->header.frame_id = world_frame_id;
    std::cout << "Loading point cloud: " << PC_ptr->size() << std::endl;
    TS_buf.clear();

    ros::Rate r(mapping_rate_hz);
    while (ros::ok())
    {
        ros::spinOnce();
        r.sleep();
        if (!strRep.compare("TS") && TS_buf.size() < TS_start) // tracking initialize
        {
            continue;
        }
        else if (!strRep.compare("EM") && event_number < Event_start)
        {
            continue;
        }

        sensor_msgs::PointCloud2::Ptr pc_to_publish(new sensor_msgs::PointCloud2);
        if (!PC_ptr->empty())
        {
            pcl::toROSMsg(*PC_ptr, *pc_to_publish);
            if (!strRep.compare("TS"))
            {
                // auto it_end = TS_buf.rbegin();
                // it_end++;
                // while (it_end != TS_buf.rend())
                // {
                //     if (getPoseAt(it_end->header.stamp, dvs_frame_id))
                //     {
                //         TS_latest_timestamp = it_end->header.stamp;
                //         std::cout << TS_latest_timestamp << std::endl;
                //         break;
                //     }
                //     it_end++;
                // }
                // if (it_end == TS_buf.rend())
                // {
                //     continue;
                // }
                auto it = TS_buf.end() - 5;
                TS_latest_timestamp = it->header.stamp;
                pc_to_publish->header.stamp = TS_latest_timestamp;
            }
            else if (!strRep.compare("EM"))
            {
                pc_to_publish->header.stamp = EM_latest_timestamp;
            }
            pc_pub.publish(pc_to_publish);
        }
    }
    return 0;
}
