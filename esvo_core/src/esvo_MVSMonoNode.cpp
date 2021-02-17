#include <esvo_core/esvo_MVSMono.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "esvo_MVSMono");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  esvo_core::esvo_MVSMono mvs(nh, nh_private);
  ros::spin();
  return 0;
}