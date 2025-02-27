
# """csci3302_lab4 controller."""
# Copyright (2025) University of Colorado Boulder
# CSCI 3302: Introduction to Robotics

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
import math
import time
import random
import copy
import numpy as np
from controller import Robot, Motor, DistanceSensor

state = "line_follower" # Change this to anything else to stay in place to test coordinate transform functions

LIDAR_SENSOR_MAX_RANGE = 3 # Meters
LIDAR_ANGLE_BINS = 21 # 21 Bins to cover the angular range of the lidar, centered at 10
LIDAR_ANGLE_RANGE = 1.5708 # 90 degrees, 1.5708 radians

# These are your pose values that you will update by solving the odometry equations
pose_x = 0.197
pose_y = 0.678
pose_theta = -np.pi

# ePuck Constants
EPUCK_AXLE_DIAMETER = 0.053 # ePuck's wheels are 53mm apart.
MAX_SPEED = 6.28

# create the Robot instance.
robot=Robot()

# get the time step of the current world.
SIM_TIMESTEP = int(robot.getBasicTimeStep())

# Initialize Motors
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

# Initialize and Enable the Ground Sensors
gsr = [0, 0, 0]
ground_sensors = [robot.getDevice('gs0'), robot.getDevice('gs1'), robot.getDevice('gs2')]
for gs in ground_sensors:
    gs.enable(SIM_TIMESTEP)

# Initialize the Display    
display = robot.getDevice("display")

# get and enable lidar 
lidar = robot.getDevice("LDS-01")
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
known_obs=set()
known_free=set()
known_path=set()
lidar_readings = []
lidar_angles = np.linspace(-LIDAR_ANGLE_RANGE/2, LIDAR_ANGLE_RANGE/2, LIDAR_ANGLE_BINS)
#### End of Part 1 #####
lidar_offsets = []
mid_index = LIDAR_ANGLE_BINS // 2  # For 21 bins, the middle index is 10.
angle_increment = LIDAR_ANGLE_RANGE / (LIDAR_ANGLE_BINS - 1)  # Step size between beams.
for i in range(LIDAR_ANGLE_BINS):
    offset = (i - mid_index) * angle_increment
    lidar_offsets.append(offset)
last_second = -1
# Main Control Loop:
i=0

map_width = display.getWidth()
map_height = display.getHeight()

def bresenham_line(x0, y0, x1, y1):
    """Return a list of pixel coordinates on the line from (x0,y0) to (x1,y1)."""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x1, y1))
    return np.array(points)

occupancy_grid = [[None for _ in range(map_width)] for _ in range(map_height)]
display.setColor(0x000000)  # Black background.
display.fillRectangle(0, 0, map_width, map_height)
display.imageSave(None,"map.png") 
while robot.step(SIM_TIMESTEP) != -1:     
    #####################################################
    #                 Sensing                           #
    #####################################################

    # Read ground sensors
    for i, gs in enumerate(ground_sensors):
        gsr[i] = gs.getValue()

    # Read Lidar           
    lidar_sensor_readings = lidar.getRangeImage() # rhos
    
    
    ##### Part 2: Turn world coordinates into map coordinates
    #
    # Come up with a way to turn the robot pose (in world coordinates)
    # into coordinates on the map. Draw a red dot using display.drawPixel()
    # where the robot moves.
    
    pixel_x = int(pose_x * map_width)
    pixel_y = int(pose_y * map_height)
    # known_path.add((pixel_x, pixel_y))
    
    ##### Part 3: Convert Lidar data into world coordinates
    #
    # Each Lidar reading has a distance rho and an angle alpha.
    # First compute the corresponding rx and ry of where the lidar
    # hits the object in the robot coordinate system. Then convert
    # rx and ry into world coordinates wx and wy. 
    # The arena is 1x1m2 and its origin is in the top left of the arena. 
    # if state=='stationary':
    free_space_lines = []
    obstacle_points = []
    robot_world_trans_matrix = np.array([
        [np.cos(pose_theta), -np.sin(pose_theta), pose_x],
        [np.sin(pose_theta),  np.cos(pose_theta), pose_y],
        [0,                   0,                  1     ]
    ])

    for i, rho in enumerate(lidar_sensor_readings):
        if np.isinf(rho):
            continue
        alpha = lidar_offsets[i]  # beam's angular offset relative to forward
        
        # Convert polar to robot-frame Cartesian coordinates:
        r_x = rho * np.sin(alpha)   # lateral offset (to right)
        r_y = rho * np.cos(alpha)   # forward offset
        
        # Form the homogeneous coordinate for the robot-frame point.
        robot_point = np.array([r_x, r_y, 1.0])
        
        # Use the transformation matrix to get world coordinates.
        world_point = robot_world_trans_matrix.dot(robot_point)
        wx, wy = world_point[0], world_point[1]
        
        # Convert to pixel coordinates (assuming a 1×1 m arena mapped to display dimensions)
        obs_pixel_x = int(wx * map_width)
        obs_pixel_y = int(wy * map_height)
        
        # Store the free space line (from robot to obstacle) and the obstacle point.
        # free_space_lines.append((pixel_x, pixel_y, obs_pixel_x, obs_pixel_y))
        # obstacle_points.append((obs_pixel_x, obs_pixel_y))
        # if (obs_pixel_x, obs_pixel_y) not in known_obs:
        #     known_obs.add((obs_pixel_x, obs_pixel_y))
        print("Obstacle detected at: ", obs_pixel_x, obs_pixel_y)

        # if (pixel_x, pixel_y, obs_pixel_x, obs_pixel_y) not in known_free:
        #     known_free.add((pixel_x, pixel_y, obs_pixel_x, obs_pixel_y))
        # print("Free space detected from: ", pixel_x, pixel_y, " to ", obs_pixel_x, obs_pixel_y)
        # line_points = bresenham_line(pixel_x, pixel_y, obs_pixel_x, obs_pixel_y)
        # Mark every cell along the beam as free space (0), except the last cell.
        # for (x, y) in line_points[:-1]:
        #     if 0 <= x < map_width and 0 <= y < map_height:
        #         occupancy_grid[y][x] = 0
        # Mark the endpoint as an obstacle (1).
        # end_x, end_y = line_points[-1]
        # if 0 <= end_x < map_width and 0 <= end_y < map_height:
        #     occupancy_grid[end_y][end_x] = 1
        if 0 <= obs_pixel_x < map_width and 0 <= obs_pixel_y < map_height:
            occupancy_grid[obs_pixel_y][obs_pixel_x] = 1
    
    ##### Part 4: Draw the obstacle and free space pixels on the map

    # display.setColor(0xFFFFFF)  # white color
    # for (start_x, start_y, end_x, end_y) in free_space_lines:
    #     display.drawLine(start_x, start_y, end_x, end_y)
    # for (start_x, start_y, end_x, end_y) in known_free:
    #     display.drawLine(start_x, start_y, end_x, end_y)

    # Draw the obstacles: blue pixels at the detected endpoints.
    # display.setColor(0x0000FF)  # blue color
    # for (px, py) in obstacle_points:
    #     display.drawPixel(px, py)
    # for (px, py) in known_obs:
    #     display.drawPixel(px, py)

    # Draw a red dot on the map at the robot's current location
    # print("Pixel coordinates: ", pixel_x, pixel_y)
    # display.setColor(0xff0000)  # Set the color to red (in hexadecimal)
    # # display.drawPixel(pixel_x, pixel_y)
    # for (px, py) in known_path:
    #     display.drawPixel(px, py)
    # current_time = robot.getTime()
    # current_second = int(current_time)
    # if i%3==0:
    # display.imageSave(None,"map.png") 
        # last_second = current_second
    occupancy_grid[pixel_y][pixel_x] = 2
    # DO NOT CHANGE THE FOLLOWING CODE (You might only add code to display robot poses)
    #####################################################
    #                 Robot controller                  #
    #####################################################

    if state == "line_follower":
            if(gsr[1]<350 and gsr[0]>400 and gsr[2] > 400):
                vL=MAX_SPEED*0.3
                vR=MAX_SPEED*0.3                
            # Checking for Start Line          
            elif(gsr[0]<500 and gsr[1]<500 and gsr[2]<500):
                vL=MAX_SPEED*0.3
                vR=MAX_SPEED*0.3
                print("Over the line!") # Feel free to uncomment thi
                for y in range(map_height):
                    for x in range(map_width):
                        cell = occupancy_grid[y][x]
                        if cell is not None:
                            if cell == 0:
                                display.setColor(0xFFFFFF)  # White for free space.
                            elif cell == 1:
                                display.setColor(0x0000FF)  # Blue for obstacles.
                            elif cell == 2:
                                 display.setColor(0xff0000)  # Red color.
                            display.drawPixel(x, y)
                # Draw the robot’s current location as a red dot.
                # display.setColor(0xff0000)  # Red color.
                # display.drawPixel(pixel_x, pixel_y)
                display.imageSave(None,"map.png") 
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
    
    EPUCK_MAX_WHEEL_SPEED = 0.11695*SIM_TIMESTEP/1000.0 
    dsr=vR/MAX_SPEED*EPUCK_MAX_WHEEL_SPEED
    dsl=vL/MAX_SPEED*EPUCK_MAX_WHEEL_SPEED
    ds=(dsr+dsl)/2.0
    
    pose_y += ds*math.cos(pose_theta)
    pose_x += ds*math.sin(pose_theta)
    pose_theta += (dsr-dsl)/EPUCK_AXLE_DIAMETER
    i+=1
    # Feel free to uncomment this for debugging
    print("X: %f Y: %f Theta: %f " % (pose_x,pose_y,pose_theta))
