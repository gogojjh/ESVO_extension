#include <esvo_core/Tracking.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "Tracking");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  esvo_core::Tracking tracker(nh, nh_private);
  ros::spin();
  return 0;
}

