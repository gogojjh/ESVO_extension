# /bin/bash

echo "Start Testing on indoor_flying1, indoor_flying3 ..."
###################################### indoor_flying1
# roslaunch esvo_core system_upenn.launch \
#     Dataset_Name:=upenn Sequence_Name:=indoor_flying1 \
#     Representation_Name:=TS tracking_rate_hz:=100 kernelSize:=5
# sleep 2

# roslaunch esvo_core system_upenn.launch \
#     Dataset_Name:=upenn Sequence_Name:=indoor_flying1 \
#     Representation_Name:=EM eventNum_EM:=2000 tracking_rate_hz:=1000 kernelSize:=5
# sleep 2

# roslaunch esvo_core system_upenn.launch \
#     Dataset_Name:=upenn Sequence_Name:=indoor_flying1 \
#     Representation_Name:=EM eventNum_EM:=3000 tracking_rate_hz:=1000 kernelSize:=5
# sleep 2

# roslaunch esvo_core system_upenn.launch \
#     Dataset_Name:=upenn Sequence_Name:=indoor_flying1 \
#     Representation_Name:=EM eventNum_EM:=4000 tracking_rate_hz:=1000 kernelSize:=5
# sleep 2

# roslaunch esvo_core system_upenn.launch \
#     Dataset_Name:=upenn Sequence_Name:=indoor_flying1 \
#     Representation_Name:=EM eventNum_EM:=5000 tracking_rate_hz:=1000 kernelSize:=5
# sleep 2

###################################### indoor_flying3
roslaunch esvo_core system_upenn.launch \
    Dataset_Name:=upenn Sequence_Name:=indoor_flying3 \
    Representation_Name:=TS tracking_rate_hz:=100 kernelSize:=5
# sleep 2

# roslaunch esvo_core system_upenn.launch \
#     Dataset_Name:=upenn Sequence_Name:=indoor_flying3 \
#     Representation_Name:=EM eventNum_EM:=2000 tracking_rate_hz:=1000 kernelSize:=5
# sleep 2

# roslaunch esvo_core system_upenn.launch \
#     Dataset_Name:=upenn Sequence_Name:=indoor_flying3 \
#     Representation_Name:=EM eventNum_EM:=3000 tracking_rate_hz:=1000 kernelSize:=5
# sleep 2

# roslaunch esvo_core system_upenn.launch \
#     Dataset_Name:=upenn Sequence_Name:=indoor_flying3 \
#     Representation_Name:=EM eventNum_EM:=4000 tracking_rate_hz:=1000 kernelSize:=5
# sleep 2

# roslaunch esvo_core system_upenn.launch \
#     Dataset_Name:=upenn Sequence_Name:=indoor_flying3 \
#     Representation_Name:=EM eventNum_EM:=5000 tracking_rate_hz:=1000 kernelSize:=5 
# sleep 2
# echo "Finish Testing !"

########################################
echo "Start Evaluation on indoor_flying1, indoor_flying3 ..."
python2 /home/jjiao/catkin_ws/src/localization/rpg_trajectory_evaluation/scripts/analyze_trajectory_single_vo.py\
    --est_types TS EM2000 EM3000 EM4000 EM5000\
    --recalculate_errors --compare /home/jjiao/ESVO_result/upenn/indoor_flying1/traj

python2 /home/jjiao/catkin_ws/src/localization/rpg_trajectory_evaluation/scripts/analyze_trajectory_single_vo.py\
    --est_types TS EM2000 EM3000 EM4000 EM5000\
    --recalculate_errors --compare /home/jjiao/ESVO_result/upenn/indoor_flying3/traj

python3 /home/jjiao/catkin_ws/src/localization/rpg_trajectory_evaluation/scripts/load_eval_results.py\
    -path /home/jjiao/ESVO_result/upenn\
    -sequence=indoor_flying1,indoor_flying3 -est_type=TS,EM2000,EM3000,EM4000,EM5000\
    -eval_type=single -err_type=ate #rpe
echo "Finish Evaluation !"

