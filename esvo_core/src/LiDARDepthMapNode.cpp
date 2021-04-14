#include <esvo_core/LiDARDepthMap.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "LiDARDepthMap");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  esvo_core::LiDARDepthMap lidarDepthMapper(nh, nh_private);
  ros::spin();
  return 0;
}

