#include <esvo_core/esvo_MonoTracking.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "esvo_MonoTracking");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  esvo_core::esvo_MonoTracking tracker(nh, nh_private);
  ros::spin();
  return 0;
}

