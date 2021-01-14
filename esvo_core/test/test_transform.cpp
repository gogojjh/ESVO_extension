#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <tf/tf.h>
#include <tf/tfMessage.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "test_transform");
    ros::NodeHandle nh;

    ros::Rate r(10);

    ros::Time t_begin = ros::Time::now();
    tf::Transformer tf_;
    for (size_t i = 0; i <= 10; i++)
    {
        if (!ros::ok())
            break;
        tf::Transform tf(
            tf::Quaternion(
                0,
                0,
                0,
                1),
            tf::Vector3(
                i * 1,
                0,
                0));
        ros::Time t_cur = ros::Time::now();
        // printf("t_cur: %fs\n", t_cur.toSec());
        tf::StampedTransform st(tf, t_cur, "world", "base_link");
        tf_.setTransform(st);
        static tf::TransformBroadcaster br;
        br.sendTransform(st);
        // ros::Time t_test = ros::Time(t_cur.toSec() - 0.3);
        // printf("t_test: %fs\n", t_test.toSec());
        // std::string *err_msg = new std::string();
        // if (!tf_.canTransform("world", "base_link", t_test, err_msg))
        // {
        //     std::cout << *err_msg << std::endl;
        //     delete err_msg;
        //     // continue;
        // }
        // else
        // {
        //     tf::StampedTransform st;
        //     tf_.lookupTransform("world", "base_link", t_test, st);
        //     tf::Quaternion q;
        //     tf::Vector3 t;
        //     t = st.getOrigin();
        //     q = st.getRotation();
        //     std::cout << t.x() << " " << t.y() << " " << t.z() << std::endl;
        // }
        r.sleep();
        // sleep(1);
    }

    ros::Time t_end = ros::Time::now();
    double dt = (t_end.toSec() - t_begin.toSec()) / 20;
    printf("t_begin: %fs\n", t_begin.toSec());
    printf("t_end: %fs\n", t_end.toSec());
    printf("dt: %fs\n", dt);
    for (size_t i = 0; i < 20; i++)
    {
        ros::Time t_cur = ros::Time::now().fromSec(t_begin.toSec() + dt * i);
        std::string *err_msg = new std::string();
        if (!tf_.canTransform("world", "base_link", t_cur, err_msg))
        {
            std::cout << *err_msg << std::endl;
            delete err_msg;
            continue;
        }
        else
        {
            tf::StampedTransform st;           
            tf_.lookupTransform("world", "base_link", t_cur, st);
            tf::Quaternion q;
            tf::Vector3 t;
            t = st.getOrigin();
            q = st.getRotation();
            std::cout << i << ": " << t.x() << " " << t.y() << " " << t.z() << std::endl;
        }
    }

    return 0;
}