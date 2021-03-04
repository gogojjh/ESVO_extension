#include <esvo_core/esvo_MonoMapping.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "esvo_MonoMapping");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  esvo_core::esvo_MonoMapping mapper(nh, nh_private);
  ros::spin();
  return 0;
}

