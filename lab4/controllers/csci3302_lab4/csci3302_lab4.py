
# """csci3302_lab4 controller."""
# Copyright (2025) University of Colorado Boulder
# CSCI 3302: Introduction to Robotics

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
import math
import time
import random
import copy
from turtle import left
import numpy as np
from controller import Robot, Motor, DistanceSensor

state="line_follower" # Change this to anything else to stay in place to test coordinate transform functions

LIDAR_SENSOR_MAX_RANGE=3 # Meters
LIDAR_ANGLE_BINS=21 # 21 Bins to cover the angular range of the lidar, centered at 10
LIDAR_ANGLE_RANGE=1.5708 # 90 degrees, 1.5708 radians

# These are your pose values that you will update by solving the odometry equations
pose_x=0.197
pose_y=0.678
pose_theta=-np.pi

# ePuck Constants
EPUCK_AXLE_DIAMETER=0.053 # ePuck's wheels are 53mm apart.
MAX_SPEED=6.28

# create the Robot instance.
robot=Robot()

# get the time step of the current world.
SIM_TIMESTEP=int(robot.getBasicTimeStep())

# Initialize Motors
leftMotor=robot.getDevice('left wheel motor')
rightMotor=robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

# Initialize and Enable the Ground Sensors
gsr=[0, 0, 0]
ground_sensors=[robot.getDevice('gs0'), robot.getDevice('gs1'), robot.getDevice('gs2')]
for gs in ground_sensors:
    gs.enable(SIM_TIMESTEP)

# Initialize the Display    
display=robot.getDevice("display")

# get and enable lidar 
lidar=robot.getDevice("LDS-01")
lidar.enable(SIM_TIMESTEP)
lidar.enablePointCloud()

##### DO NOT MODIFY ANY CODE ABOVE THIS #####

##### Part 1: Setup Data structures
#
# Create an empty list for your lidar sensor readings here,
# as well as an array that contains the angles of each ray 
# in radians. The total field of view is LIDAR_ANGLE_RANGE,
# and there are LIDAR_ANGLE_BINS. An easy way to generate the
# array that contains all the angles is to use linspace from
# the numpy package.
lidar_offsets=[]
mid_index=LIDAR_ANGLE_BINS // 2  # For 21 bins, the middle index is 10.
angle_increment=LIDAR_ANGLE_RANGE / (LIDAR_ANGLE_BINS - 1)  # Step size between beams.
for i in range(LIDAR_ANGLE_BINS):
    offset=(i - mid_index) * angle_increment
    lidar_offsets.append(offset)
#### End of Part 1 #####
map_width=display.getWidth()
map_height=display.getHeight()

def bresenham_line(x0, y0, x1, y1):
    x0, y0, x1, y1=int(x0), int(y0), int(x1), int(y1)
    dx=abs(x1 - x0)
    dy=abs(y1 - y0)
    points=np.empty((max(dx, dy) + 1, 2), dtype=np.int32)
    x, y=x0, y0
    sx=1 if x0<x1 else -1
    sy=1 if y0<y1 else -1
    if dx>dy: err=dx / 2.0
    else: err=dy / 2.0
    i=0
    while True:
        points[i]=(x, y)
        i += 1
        if x==x1 and y==y1: break
        if dx>dy:
            err -= dy
            if err<0:
                y += sy
                err += dx
            x += sx
        else:
            err -= dx
            if err<0:
                x += sx
                err += dy
            y += sy
    return points[:i]

occupancy_grid=np.full((map_height, map_width), fill_value=-1, dtype=np.int8)
display.setColor(0x000000)  # Black background.
display.fillRectangle(0, 0, map_width, map_height)
display.imageSave(None,"map.png") 
black_line_start_time=None
start_line_triggered=False

while robot.step(SIM_TIMESTEP) != -1:     
    #####################################################
    #                 Sensing                           #
    #####################################################

    # Read ground sensors
    for i, gs in enumerate(ground_sensors):
        gsr[i]=gs.getValue()

    # Read Lidar           
    lidar_sensor_readings=lidar.getRangeImage() # rhos
    ##### Part 2: Turn world coordinates into map coordinates
    #
    # Come up with a way to turn the robot pose (in world coordinates)
    # into coordinates on the map. Draw a red dot using display.drawPixel()
    # where the robot moves.
    pixel_x=int(pose_x * map_width)
    pixel_y=int(pose_y * map_height)
    ##### Part 3: Convert Lidar data into world coordinates
    #
    # Each Lidar reading has a distance rho and an angle alpha.
    # First compute the corresponding rx and ry of where the lidar
    # hits the object in the robot coordinate system. Then convert
    # rx and ry into world coordinates wx and wy. 
    # The arena is 1x1m2 and its origin is in the top left of the arena. 
    # Build the transformation matrix using the corrected formulation:
    robot_world_trans_matrix=np.array([
        [ np.cos(pose_theta),  np.sin(pose_theta), pose_x],
        [-np.sin(pose_theta),  np.cos(pose_theta), pose_y],
        [0,                   0,                  1     ]
    ])
    for i, rho in enumerate(lidar_sensor_readings):
        if np.isinf(rho): continue
        alpha=lidar_offsets[i]
        r_x=-rho * np.sin(alpha)
        r_y= rho * np.cos(alpha)
        world_point=robot_world_trans_matrix.dot(np.array([r_x, r_y, 1.0]))
        obs_pixel_x=int(world_point[0] * map_width)
        obs_pixel_y=int(world_point[1] * map_height)
        line_points=bresenham_line(pixel_x, pixel_y, obs_pixel_x, obs_pixel_y)
        obs_points=line_points[-1]
        line_points=line_points[:-1]
        mask=(line_points[:,0] >= 0) & (line_points[:,0]<map_width) & (line_points[:,1] >= 0) & (line_points[:,1]<map_height)
        valid_points=line_points[mask]
        if valid_points.size>0:
            indices_y=valid_points[:,1]
            indices_x=valid_points[:,0]
            free_mask=occupancy_grid[indices_y, indices_x] != 2  # Only update if not already marked 2
            occupancy_grid[indices_y[free_mask], indices_x[free_mask]]=0
        end_x, end_y=obs_points
        if 0 <= end_x<map_width and 0 <= end_y<map_height and occupancy_grid[end_y, end_x] != 2: occupancy_grid[end_y, end_x]=1
    occupancy_grid[pixel_y,pixel_x]=2
    
    ##### Part 4: Draw the obstacle and free space pixels on the map
    all_black=all(gs.getValue()<400 for gs in ground_sensors)
    if all_black:
        if black_line_start_time is None: black_line_start_time=robot.getTime()
        elif robot.getTime() - black_line_start_time>0.2 and not start_line_triggered:
            print("Start line detected, saving image")
            for y in range(map_height):
                for x in range(map_width):
                    cell=occupancy_grid[y][x]
                    if cell is not None:
                        if cell==0: display.setColor(0xFFFFFF)
                        elif cell==1: display.setColor(0x0000FF)
                        elif cell==2: display.setColor(0xff0000)
                        elif cell==-1: display.setColor(0x000000)
                        display.drawPixel(x, y)
            display.imageSave(None, "map.png")
            start_line_triggered=True
    else:
        black_line_start_time=None
        start_line_triggered=False
    
    # DO NOT CHANGE THE FOLLOWING CODE (You might only add code to display robot poses)
    #####################################################
    #                 Robot controller                  #
    #####################################################

    if state=="line_follower":
        if(gsr[1]<350 and gsr[0]>400 and gsr[2]>400):
            vL=MAX_SPEED*0.3
            vR=MAX_SPEED*0.3                
        # Checking for Start Line          
        elif(gsr[0]<500 and gsr[1]<500 and gsr[2]<500):
            vL=MAX_SPEED*0.3
            vR=MAX_SPEED*0.3
            # print("Over the line!") # Feel free to uncomment thi
        elif(gsr[2]<650): # turn right
            vL=0.2*MAX_SPEED
            vR=-0.05*MAX_SPEED
        elif(gsr[0]<650): # turn left
            vL=-0.05*MAX_SPEED
            vR=0.2*MAX_SPEED
    else:
        # Stationary State
        vL=0
        vR=0   
    
    leftMotor.setVelocity(vL)
    rightMotor.setVelocity(vR)
    
    #####################################################
    #                    Odometry                       #
    #####################################################
    
    EPUCK_MAX_WHEEL_SPEED=0.11695*SIM_TIMESTEP/1000.0 
    dsr=vR/MAX_SPEED*EPUCK_MAX_WHEEL_SPEED
    dsl=vL/MAX_SPEED*EPUCK_MAX_WHEEL_SPEED
    ds=(dsr+dsl)/2.0
    
    pose_y += ds*math.cos(pose_theta)
    pose_x += ds*math.sin(pose_theta)
    pose_theta += (dsr-dsl)/EPUCK_AXLE_DIAMETER
    
    # Feel free to uncomment this for debugging
    # print("X: %f Y: %f Theta: %f " % (pose_x,pose_y,pose_theta))
