# -*- coding: utf-8 -*- 
"""
Project: IMU_Path_Visualisation
Creator: Administrator
Create time: 2020-01-13 11:23
IDE: PyCharm
Introduction: call class Vis by using input files position and euler to generate path data and gesture data
              and generate VTP files of them for paraview.
"""
from IMU_Path_Vis import IMU_Vis

if __name__ == "__main__":
    pos_file = "input_file\\pos-algo0_0_hig.csv"
    eul_file = "input_file\\att_euler-algo0_0.csv"
    Vis = IMU_Vis.Vis(pos_file=pos_file, eul_file=eul_file)
    Vis.gen_gesture_on_path(num=10)
    data = Vis.get_path_gesture_data()
    Vis.gen_vtk_for_path_gesture("path", "output_file\\point_data.vtp")
    Vis.gen_vtk_for_path_gesture("gesture", "output_file\\gesture_data.vtp")





