# /bin/bash

echo "Start Testing ..."
###################################### rpg_monitor
roslaunch esvo_core system_rpg_stereo.launch \
    Dataset_Name:=rpg_stereo Sequence_Name:=rpg_monitor \
    Representation_Name:=TS tracking_rate_hz:=100
sleep 2

roslaunch esvo_core system_rpg_stereo.launch \
    Dataset_Name:=rpg_stereo Sequence_Name:=rpg_monitor \
    Representation_Name:=EM eventNum_EM:=2000 tracking_rate_hz:=1000
sleep 2

roslaunch esvo_core system_rpg_stereo.launch \
    Dataset_Name:=rpg_stereo Sequence_Name:=rpg_monitor \
    Representation_Name:=EM eventNum_EM:=3000 tracking_rate_hz:=1000
sleep 2

roslaunch esvo_core system_rpg_stereo.launch \
    Dataset_Name:=rpg_stereo Sequence_Name:=rpg_monitor \
    Representation_Name:=EM eventNum_EM:=4000 tracking_rate_hz:=1000
sleep 2

roslaunch esvo_core system_rpg_stereo.launch \
    Dataset_Name:=rpg_stereo Sequence_Name:=rpg_monitor \
    Representation_Name:=EM eventNum_EM:=5000 tracking_rate_hz:=1000    
sleep 2

###################################### rpg_bin
roslaunch esvo_core system_rpg_stereo.launch \
    Dataset_Name:=rpg_stereo Sequence_Name:=rpg_bin \
    Representation_Name:=TS tracking_rate_hz:=100
sleep 2

roslaunch esvo_core system_rpg_stereo.launch \
    Dataset_Name:=rpg_stereo Sequence_Name:=rpg_bin \
    Representation_Name:=EM eventNum_EM:=2000 tracking_rate_hz:=1000
sleep 2

roslaunch esvo_core system_rpg_stereo.launch \
    Dataset_Name:=rpg_stereo Sequence_Name:=rpg_bin \
    Representation_Name:=EM eventNum_EM:=3000 tracking_rate_hz:=1000
sleep 2

roslaunch esvo_core system_rpg_stereo.launch \
    Dataset_Name:=rpg_stereo Sequence_Name:=rpg_bin \
    Representation_Name:=EM eventNum_EM:=4000 tracking_rate_hz:=1000
sleep 2

roslaunch esvo_core system_rpg_stereo.launch \
    Dataset_Name:=rpg_stereo Sequence_Name:=rpg_bin \
    Representation_Name:=EM eventNum_EM:=5000 tracking_rate_hz:=1000    
sleep 2

###################################### rpg_desk
roslaunch esvo_core system_rpg_stereo.launch \
    Dataset_Name:=rpg_stereo Sequence_Name:=rpg_desk \
    Representation_Name:=TS tracking_rate_hz:=100
sleep 2

roslaunch esvo_core system_rpg_stereo.launch \
    Dataset_Name:=rpg_stereo Sequence_Name:=rpg_desk \
    Representation_Name:=EM eventNum_EM:=2000 tracking_rate_hz:=1000
sleep 2

roslaunch esvo_core system_rpg_stereo.launch \
    Dataset_Name:=rpg_stereo Sequence_Name:=rpg_desk \
    Representation_Name:=EM eventNum_EM:=3000 tracking_rate_hz:=1000
sleep 2

roslaunch esvo_core system_rpg_stereo.launch \
    Dataset_Name:=rpg_stereo Sequence_Name:=rpg_desk \
    Representation_Name:=EM eventNum_EM:=4000 tracking_rate_hz:=1000
sleep 2

roslaunch esvo_core system_rpg_stereo.launch \
    Dataset_Name:=rpg_stereo Sequence_Name:=rpg_desk \
    Representation_Name:=EM eventNum_EM:=5000 tracking_rate_hz:=1000    
sleep 2

###################################### rpg_box
roslaunch esvo_core system_rpg_stereo.launch \
    Dataset_Name:=rpg_stereo Sequence_Name:=rpg_box \
    Representation_Name:=TS tracking_rate_hz:=100
sleep 2

roslaunch esvo_core system_rpg_stereo.launch \
    Dataset_Name:=rpg_stereo Sequence_Name:=rpg_box \
    Representation_Name:=EM eventNum_EM:=2000 tracking_rate_hz:=1000
sleep 2

roslaunch esvo_core system_rpg_stereo.launch \
    Dataset_Name:=rpg_stereo Sequence_Name:=rpg_box \
    Representation_Name:=EM eventNum_EM:=3000 tracking_rate_hz:=1000
sleep 2

roslaunch esvo_core system_rpg_stereo.launch \
    Dataset_Name:=rpg_stereo Sequence_Name:=rpg_box \
    Representation_Name:=EM eventNum_EM:=4000 tracking_rate_hz:=1000
sleep 2

roslaunch esvo_core system_rpg_stereo.launch \
    Dataset_Name:=rpg_stereo Sequence_Name:=rpg_box \
    Representation_Name:=EM eventNum_EM:=5000 tracking_rate_hz:=1000    
sleep 2
echo "Finish Testing !"

########################################
echo "Start Evaluation ..."
python2 /home/jjiao/catkin_ws/src/localization/rpg_trajectory_evaluation/scripts/analyze_trajectory_single_vo.py 
    --est_types TS EM2000 EM3000 EM4000 EM5000 \
    --recalculate_errors --compare /home/jjiao/ESVO_result/rpg_stereo/rpg_monitor/traj

python2 /home/jjiao/catkin_ws/src/localization/rpg_trajectory_evaluation/scripts/analyze_trajectory_single_vo.py 
    --est_types TS EM2000 EM3000 EM4000 EM5000 \
    --recalculate_errors --compare /home/jjiao/ESVO_result/rpg_stereo/rpg_bin/traj

python2 /home/jjiao/catkin_ws/src/localization/rpg_trajectory_evaluation/scripts/analyze_trajectory_single_vo.py 
    --est_types TS EM2000 EM3000 EM4000 EM5000 \
    --recalculate_errors --compare /home/jjiao/ESVO_result/rpg_stereo/rpg_desk/traj

python2 /home/jjiao/catkin_ws/src/localization/rpg_trajectory_evaluation/scripts/analyze_trajectory_single_vo.py 
    --est_types TS EM2000 EM3000 EM4000 EM5000 \
    --recalculate_errors --compare /home/jjiao/ESVO_result/rpg_stereo/rpg_box/traj        

python scripts/load_eval_results.py -path results/rpg_stereo \  
    -sequence=rpg_monitor,rpg_bin,rpg_desk,rpg_box -est_type=TS,EM \
    -eval_type=single -err_type=ate,rpe
echo "Finish Evaluation !"

