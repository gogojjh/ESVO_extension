# ! /usr/bin/python2
# Usage: python run_esvo.py -dataset=rpg_stereo -sequence=rpg_monitor,rpg_box
#           -representation=TS,EM,TSEM -eventnum=2000,3000,4000,5000 -trials=10 -program=eval
# or
# python script/run_esvo.py -dataset=rpg_stereo,upenn -sequence=rpg_desk,rpg_bin,rpg_box,indoor_flying1,indoor_flying3 \
#     -representation=TS,EM,TSEM -eventnum=2000,3000,4000,5000 -trials=10 -program=eval
import os
import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import fn_constants as fn

dataset = []
sequence = []
representation = []
eventnum = []
trials = 1
program = []
example_py_command = 'python run_esvo.py -dataset=rpg_stereo -sequence=rpg_monitor,rpg_box \
                    -representation=TS,EM,TSEM -eventnum=2000,3000,4000,5000 -trials=10'
dir_esvo_result = '~/ESVO_result'
dir_eval_code = '~/catkin_ws/src/localization/rpg_trajectory_evaluation'

def run_esvo(dataset, sequence, representation, eventnum, trials):
    os.system('rm /tmp/run_esvo_output')
# 
    os.system('echo \"Start Testing\"')
    for ds in dataset:
        for seq in sequence:
            if (not seq in fn.seqsToDatasetsMapping[ds]):
                continue
            command = 'mkdir -p {}/{}/{}/traj'.format(dir_esvo_result, ds, seq)
            os.system(command)
            print(command)            
            command = 'mkdir -p {}/{}/{}/time'.format(dir_esvo_result, ds, seq)
            os.system(command)
            print(command)
            for rep in representation:
                if (rep == 'EM'):
                    tk_rate = 1000
                    eventnum_tmp = eventnum[:]
                else:
                    tk_rate = 100
                    eventnum_tmp = [4000]
                for en in eventnum_tmp:
                    if (rep == 'EM'):
                        est_type = '{}{}'.format(rep, en)
                    else:
                        est_type = rep
                    for tr in range(1, trials + 1):
                        # print('trials: {}'.format(tr))
                        command = 'roslaunch esvo_core system_{}.launch Dataset_Name:={} Sequence_Name:={} Representation_Name:={} eventNum_EM:={} tracking_rate_hz:={}'\
                                .format(ds, ds, seq, rep, en, tk_rate)
                        print('Testing Command: {}'.format(command))
                        # os.system(command + ' > /tmp/run_esvo_output')
                        os.system(command)
                        if (trials > 1):
                            command = 'mv {}/{}/{}/traj/{}_traj_estimate.txt {}/{}/{}/traj/{}_traj_estimate{}.txt'\
                                .format(dir_esvo_result, ds, seq, est_type, dir_esvo_result, ds, seq, est_type, tr - 1)
                            os.system(command)
                        command = 'sleep 2'
                        os.system(command)
    os.system('echo \"Finish Testing!\n\"')

def eval(dataset, sequence, representation, eventnum, trials):
    os.system('rm /tmp/eval_output')
#
    os.system('echo \"Start Evaluation on Sequences: {} ...\"'.format(sequence))
    for ds in dataset:
        for seq in sequence:
            if (not seq in fn.seqsToDatasetsMapping[ds]):
                continue
            est_type = ''
            for rep in representation:
                if (rep == 'EM'):
                    eventnum_tmp = eventnum[:]
                    for en in eventnum_tmp:
                        est_type = est_type + '{}{} '.format(rep, en)
                else:
                    est_type = est_type + rep + ' '
            # print(est_type)
            if (trials > 1):
                command = 'python2 {}/scripts/analyze_trajectory_single_vo.py \
                    --est_types {} --recalculate_errors \
                    --compare {}/{}/{}/traj --mul_trials={}'\
                        .format(dir_eval_code, est_type, dir_esvo_result, ds, seq, trials)
            else:
                command = 'python2 {}/scripts/analyze_trajectory_single_vo.py \
                    --est_types {} --recalculate_errors \
                    --compare {}/{}/{}/traj'\
                        .format(dir_eval_code, est_type, dir_esvo_result, ds, seq)
            print('Evaluation Command: {}'.format(command))
            os.system(command + ' > /tmp/eval_output')
    os.system('echo \"Finish Evaluation!\n\"')

def load_results(dataset, sequence, representation, eventnum, trials):
    os.system('rm /tmp/load_results_output')    
#    
    os.system('echo \"Start Summarizing Results\"')
    for ds in dataset:
        seq = ','.join([s for s in sequence if s in fn.seqsToDatasetsMapping[ds]])
        if (seq == []):
            continue
        est_type = ''
        for rep in representation:
            if (rep == 'EM'):
                eventnum_tmp = eventnum[:]
                for en in eventnum_tmp:
                    est_type = est_type + '{}{},'.format(rep, en)
            else:
                est_type = est_type + rep + ','
        est_type = est_type[:-1]
        if (trials > 1):
            command = 'python3 {}/scripts/load_eval_results.py -path {}/{} -sequence={} -est_type={} -eval_type=mc -err_type=ate'\
                .format(dir_eval_code, dir_esvo_result, ds, seq, est_type)
        else:
            command = 'python3 {}/scripts/load_eval_results.py -path {}/{} -sequence={} -est_type={} -eval_type=single -err_type=ate'\
                .format(dir_eval_code, dir_esvo_result, ds, seq, est_type)
        print('Load Results Command: {}\n'.format(command))
        os.system(command)
        os.system(command + ' > /tmp/load_results_output')
    os.system('echo \"Finish Summarizing Results\"')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run batch evaluation. Example: {}'.format(example_py_command))
    parser.add_argument('-dataset', help='rpg_stereo, upenn, simu')
    parser.add_argument('-sequence', help='rpg_monitor, rpg_box, ')
    parser.add_argument('-representation', help='TS, EM, TSEM')
    parser.add_argument('-eventnum', help='2000, 30000, 4000, 5000, ')
    parser.add_argument('-trials', type=int, help='1, 2, 3, ')
    parser.add_argument('-program', help='eval or load_result')
    args = parser.parse_args()

    dataset = args.dataset.split(',')
    sequence = args.sequence.split(',')
    representation = args.representation.split(',')
    eventnum = [int(en) for en in args.eventnum.split(',')]
    trials = args.trials
    program = args.program.split(',')
    print('Input dataset: {}'.format(dataset))
    print('Input sequence: {}'.format(sequence))
    print('Input representation: {}'.format(representation))
    print('Input eventnum: {}'.format(eventnum))
    print('Default dataset sequence mapping: {}'.format(fn.seqsToDatasetsMapping))
    print('Trials: {}'.format(trials))
    print('Program: {}'.format(program))
    if (dataset != [] and sequence != [] and representation != [] and eventnum != [] and trials != None and program != []):
        for pro in program:
            if (pro == 'run'):
                run_esvo(dataset, sequence, representation, eventnum, trials)
            elif (pro == 'eval'):
                eval(dataset, sequence, representation, eventnum, trials)
            elif (pro == 'load_result'):
                load_results(dataset, sequence, representation, eventnum, trials)
            else:
                print('{} wrong, please select proper programs!'.format(pro))


