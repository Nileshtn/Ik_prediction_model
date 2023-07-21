import bpy
import math
import pandas
import os
import time

x_list = []
y_list = []
link_01_rot_list = []
link_02_rot_list = []
link_03_rot_list = []
link_01 = bpy.data.objects["link_1_rotation"]
link_02 = bpy.data.objects["link_2_rotation"]
link_03 = bpy.data.objects["link_3_rotation"]
end_effector = bpy.data.objects["endeffector"]
ik_control = bpy.data.objects["ik_contorl_empty"]

def get_rotation(link_01, link_02, link_03):
    link_01_rot = math.degrees(link_01.matrix_local.to_euler().x)
    if link_01_rot < 0:
        link_01_rot = link_01_rot + 360
    link_02_rot = math.degrees(link_02.matrix_local.to_euler().x)
    link_02_rot =link_02_rot + 180
    link_03_rot = math.degrees(link_03.matrix_local.to_euler().x)
    link_03_rot = link_03_rot + 180

    return link_01_rot,link_02_rot, link_03_rot

for x in range(-30,30):
    for y in range(-30,30):
        ik_control.location[1] = x/5              
        ik_control.location[2] = y/5
        bpy.context.view_layer.update()
        link_01_rot,link_02_rot, link_03_rot = get_rotation(link_01, link_02, link_03)
        
        print(link_01_rot, link_02_rot, link_03_rot)

        x_list.append(x/10)
        y_list.append(y/10)
        link_01_rot_list.append(link_01_rot)
        link_02_rot_list.append(link_02_rot)
        link_03_rot_list.append(link_03_rot)
    
    
dict = {'end_effector_x' : x_list,
        'end_effector_y' : y_list,
        'link_01_rotation' : link_01_rot_list,
        'link_02_rotation' : link_02_rot_list,
        'link_03_rotation' : link_03_rot_list
        }
        
        
df = pandas.DataFrame(dict)

df.to_csv(r"D:\python\PYTORCH\blender_models\ik_blender\ik_test_data.csv")