#include <esvo_core/estimator.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "estimator");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  esvo_core::esvo_Mapping mapper(nh, nh_private);
  ros::spin();
  return 0;
}

