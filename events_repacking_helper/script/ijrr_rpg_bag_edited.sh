#!/bin/bash
set -e
if [[ $1 == "" ]];then
    echo -e "--------------------Deployment script------------------"
    echo -e "One argument is needed. Usage: \n"
    echo -e "   ./bag_edited.sh <folder_name_of_bag> <name_of_bag> <name_of_camera> \n"
    echo -e "Example: \n"
    echo -e "   ./bag_edited.sh /Monster/dataset/event_camera/ijrr_rpg_dataset slider_depth dvs \n"
    echo -e "exiting"
    echo -e "------------------------------------------------------------------"
    exit 1
fi
FOLDER_NAME=$1
BAG_NAME=$2
CAM_NAME=$3

echo -e "Start editing bags"
rosrun events_repacking_helper EventMessageEditor_Mono \
    $FOLDER_NAME/$BAG_NAME.bag $FOLDER_NAME/$BAG_NAME.bag.events /$CAM_NAME/events

python extract_topics.py \
    $FOLDER_NAME/$BAG_NAME.bag $FOLDER_NAME/$BAG_NAME.bag.extracted \
    /$CAM_NAME/camera_info /$CAM_NAME/image_raw /optitrack/davis

python merge.py \
    $FOLDER_NAME/$BAG_NAME.bag.events $FOLDER_NAME/$BAG_NAME.bag.extracted \
    --output $FOLDER_NAME/$BAG_NAME\_edited.bag
echo -e "Finish editing bags"

rm $FOLDER_NAME/$BAG_NAME.bag.events 
rm $FOLDER_NAME/$BAG_NAME.bag.extracted
echo -e "Removing temporary files"