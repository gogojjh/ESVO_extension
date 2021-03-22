#!/bin/bash
set -e
if [[ $1 == "" ]];then
    echo -e "--------------------Deployment script------------------"
    echo -e "One argument is needed. Usage: \n"
    echo -e "   ./bag_edited.sh <folder_name_of_bag> <name_of_bag> <name_of_camera1> <name_of_camera2> ...\n"
    echo -e "Example: \n"
    echo -e "   ./bag_edited.sh /Monster/dataset/event_camera/ijrr_rpg_dataset slider_depth dvs \n"
    echo -e "exiting"
    echo -e "------------------------------------------------------------------"
    exit 1
fi
FOLDER_NAME=$1
BAG_NAME=$2
CAM_NAME1=$3
CAM_NAME2=$4

echo -e "Start editing bags"
rosrun events_repacking_helper EventMessageEditor \
    $FOLDER_NAME/$BAG_NAME.bag $FOLDER_NAME/$BAG_NAME.bag.events /$CAM_NAME1/events /$CAM_NAME2/events

python extract_topics.py \
    $FOLDER_NAME/$BAG_NAME.bag $FOLDER_NAME/$BAG_NAME.bag.extracted \
    /$CAM_NAME1/camera_info /$CAM_NAME1/depthmap /$CAM_NAME1/image_corrupted \
    /$CAM_NAME1/image_raw /$CAM_NAME1/optic_flow /$CAM_NAME1/pointcloud \
    /$CAM_NAME1/pose /$CAM_NAME1/twist \
    /$CAM_NAME2/camera_info /$CAM_NAME2/depthmap /$CAM_NAME2/image_corrupted \
    /$CAM_NAME2/image_raw /$CAM_NAME2/optic_flow /$CAM_NAME2/pointcloud \
    /$CAM_NAME2/pose /$CAM_NAME2/twist \
    /imu

python merge.py \
    $FOLDER_NAME/$BAG_NAME.bag.events $FOLDER_NAME/$BAG_NAME.bag.extracted \
    --output $FOLDER_NAME/$BAG_NAME\_edited.bag
echo -e "Finish editing bags"

rm $FOLDER_NAME/$BAG_NAME.bag.events 
rm $FOLDER_NAME/$BAG_NAME.bag.extracted
echo -e "Removing temporary files"