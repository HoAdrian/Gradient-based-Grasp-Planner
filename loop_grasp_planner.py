import os
import numpy as np
import timeit
import sys
sys.path.append("../")
#from utils.constants import OBJECT_NAMES
from utils.miscellaneous_utils import print_color

STATIC_DATA_PATH = "/home/shinghei/Downloads/bao_data/static_data_original"
#DATA_RECORDING_PATH = os.path.join("/home/shinghei/Downloads/shinghei_stuff", "all_6polygon_data")

if __name__=="__main__":
    start_time = timeit.default_timer() 
    force = 15
    young_modulus = 5
    num_grasp_samples = 50
    grasp_planner_root_data_path = "/home/shinghei/Downloads/grasp_planner_data"
    vis=0
    #is_maximize_deformation = 1
    stress_net_model_path = "/home/shinghei/Downloads/shinghei_stuff/all_6polygon_open_gripper/epoch_193"
    num_epochs = 150


    for object_name in [f"6polygon0{j}" for j in [4]]:
        print_color(f"===================== Started {object_name} grasp planner data generation")
        os.system(f"python process_grasp_planner_data.py --num_samples={num_grasp_samples} --object_name={object_name} --grasp_planner_data_root_path={grasp_planner_root_data_path} --stress_net_model_path={stress_net_model_path} --young_modulus={young_modulus} --vis={vis}")

        for is_maximize_deformation in [0,1]:
            print_color(f"===================== Started {object_name} grasp planner training; is_maximize_deformation: {is_maximize_deformation==1}")
            for sample_idx in range(num_grasp_samples):
                os.system(f"python train_grasp_planner.py --init_grasp_idx={sample_idx} --is_max_deformation={is_maximize_deformation} --object_name={object_name} --grasp_planner_data_root_path={grasp_planner_root_data_path} --stress_net_model_path={stress_net_model_path} --num_epochs={num_epochs} --force={force} --young_modulus={young_modulus}")

        for is_maximize_deformation in [0,1]:
            print_color(f"===================== Started {object_name} grasp planner visualization; is_maximize_deformation: {is_maximize_deformation==1}")
            os.system(f"python visualize_grasp_pose.py --num_samples={num_grasp_samples} --is_max_deformation={is_maximize_deformation} --object_name={object_name} --grasp_planner_data_root_path={grasp_planner_root_data_path} --stress_net_model_path={stress_net_model_path} --last_epoch={num_epochs} --force={force} --young_modulus={young_modulus}")
            
        
    print_color(f"DONE! You burned {(timeit.default_timer() - start_time)/3600:.2f} trees" )