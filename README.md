# IMU_Path_Visualisation
    This project is to read IMU GPS position(LLA) data and Euler(yaw,pitch, roll) data and convert them to ECEF position data and gesture data and then generate the related path VTP file and gesture VTP file for paraview and then visaulize those data via plotly Dash.
    In IMU_Path_Vis folder, IMU_Vis.py is the class to read input files IMU GPS position(LLA) data and Euler(yaw,pitch, roll) data and convert them to ECEF position data and gesture data and then generate the related path VTP file and gesture VTP file for paraview.
    vis_main.py is the main enter program to call class Vis and generate VTP files.
    plotly_dash_gesture_on_path.py is the Dash app to call class Vis to get path and gesture data and visualize those data in plotly Dash.

## About the app plotly_dash_gesture_on_path.py

### Build with
    [Dash](https://dash.plot.ly/) - Main server and interactive components
    [Plotly Python](https://plot.ly/python/) - Used to create the interactive plots
    
### Screentshots
The following are screenshots for the app in this repo:
![picture1](https://github.com/LizzieDeng/IMU_Path_Visualisation/blob/master/Screenshots/picture1.png)
![picture2](https://github.com/LizzieDeng/IMU_Path_Visualisation/blob/master/Screenshots/picture2.png)
