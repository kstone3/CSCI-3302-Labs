"""lab5 controller."""
from os import path
from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard
import math
import numpy as np
from matplotlib import pyplot as plt
import heapq
from scipy.signal import convolve2d # Uncomment if you want to use something else for finding the configuration space

MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12

LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 2.75 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)


##### vvv [Begin] Do Not Modify vvv #####

# create the Robot instance.
robot = Robot()
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint")

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.09, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf')
robot_parts=[]

for i in range(N_PARTS):
    robot_parts.append(robot.getDevice(part_names[i]))
    robot_parts[i].setPosition(float(target_pos[i]))
    robot_parts[i].setVelocity(robot_parts[i].getMaxVelocity() / 2.0)

# The Tiago robot has a couple more sensors than the e-Puck
# Some of them are mentioned below. We will use its LiDAR for Lab 5

range = robot.getDevice('range-finder')
range.enable(timestep)
camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# We are using a GPS and compass to disentangle mapping and localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# We are using a keyboard to remote control the robot
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# The display is used to display the map. We are using 360x360 pixels to
# map the 12x12m2 apartment
display = robot.getDevice("display")

# Odometry
pose_x     = 0
pose_y     = 0
pose_theta = 0

vL = 0
vR = 0

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis

# map = None
##### ^^^ [End] Do Not Modify ^^^ #####

##################### IMPORTANT #####################
# Set the mode here. Please change to 'autonomous' before submission
mode = 'manual' # Part 1.1: manual mode
# mode = 'planner'
# mode = 'autonomous'
# mode = 'picknplace'



###################
#
# Planner
#
###################
if mode == 'planner':
    # Part 2.3: Provide start and end in world coordinate frame and convert it to map's frame
    start_w = None # (Pose_X, Pose_Y) in meters
    end_w = None # (Pose_X, Pose_Y) in meters

    # Convert the start_w and end_w from the webots coordinate frame into the map frame
    start = None # (x, y) in 360x360 map
    end = None # (x, y) in 360x360 map

    # Part 2.3: Implement A* or Dijkstra's Algorithm to find a path
    def path_planner(map, start, end):
        '''
        :param map: A 2D numpy array of size 360x360 representing the world's cspace with 0 as free space and 1 as obstacle
        :param start: A tuple of indices representing the start cell in the map
        :param end: A tuple of indices representing the end cell in the map
        :return: A list of tuples as a path from the given start to the given end in the given maze
        '''
        if map[start[0], start[1]] != 0 or map[end[0], end[1]] != 0:
            return []
        
        def heuristic(a, b):
            return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

        def get_neighbors(cell):
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx = cell[0] + dx
                    ny = cell[1] + dy
                    if 0 <= nx < map.shape[0] and 0 <= ny < map.shape[1] and map[nx, ny] == 0:
                        neighbors.append((nx, ny))
            return neighbors

        def move_cost(current, neighbor):
            return math.sqrt(2) if current[0] != neighbor[0] and current[1] != neighbor[1] else 1

        open = []
        heapq.heappush(open, (heuristic(start, end), 0, start))
        prev_location = {}
        current_cost = {start: 0}
        
        while open:
            current_priority, cost, current = heapq.heappop(open)
            if current == end:
                path = []
                while current != start:
                    path.append(current)
                    current = prev_location[current]
                path.append(start)
                path.reverse()
                return path
            
            for neighbor in get_neighbors(current):
                new_cost=current_cost[current] + move_cost(current, neighbor)
                if neighbor not in current_cost or new_cost<current_cost[neighbor]:
                    current_cost[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, end)
                    heapq.heappush(open, (priority, new_cost, neighbor))
                    prev_location[neighbor] = current
        return []

    # Part 2.1: Load map (map.npy) from disk and visualize it
    map = np.load("map.npy")
    plt.imshow(map)
    plt.show()

    # Part 2.2: Compute an approximation of the “configuration space”
    configured_map = np.copy(map)
    obstacle_detected = np.argwhere(map>0)
    footprint_radius = 8
    for x, y in obstacle_detected:
        x_min, x_max = max(0, x - footprint_radius), min(map[0], x + footprint_radius)
        y_min, y_max = max(0, y - footprint_radius), min(map[1], y + footprint_radius)

        configured_map[x_min:x_max, y_min:y_max] = 1
    plt.show(np.fliplr(configured_map), cmap = "gray")
    np.save("configured_map.npy", configured_map)
    # Part 2.3 continuation: Call path_planner
    start_world_coords = (pose_x, pose_y)
    end_world_coords = (pose_x + 1, pose_y + 1) # Replace with actual end coordinates
    path = path_planner(configured_map, start, end)

    # Part 2.4: Turn paths into waypoints and save on disk as path.npy and visualize it
    def convert_to_world(path):
        return [(x - 300 / 30, y - 390 / 30) for x, y in path]
    
    waypoints = []

    if path:
        waypoints = convert_to_world(path)
        np.save("path.npy", np.array(waypoints))

######################
#
# Map Initialization
#
######################

# Part 1.2: Map Initialization

# Initialize your map data structure here as a 2D floating point array
map = np.zeros(shape=[361,361])
waypoints = []

if mode == 'autonomous':
    # Part 3.1: Load path from disk and visualize it
    waypoints = [] # Replace with code to load your path

state = 0 # use this to iterate through your path

if mode == 'picknplace':
    # Part 4: Use the function calls from lab5_joints using the comments provided there
    ## use path_planning to generate paths
    ## do not change start_ws and end_ws below
    start_ws = [(3.7, 5.7)]
    end_ws = [(10.0, 9.3)]
    pass

while robot.step(timestep) != -1 and mode != 'planner':

    ###################
    #
    # Mapping
    #
    ###################

    ################ v [Begin] Do not modify v ##################
    # Ground truth pose
    pose_x = gps.getValues()[0]
    pose_y = gps.getValues()[1]
    
    n = compass.getValues()
    rad = -((math.atan2(n[0], n[2]))-1.5708)
    pose_theta = rad

    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]

    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]

        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue

        # The Webots coordinate system doesn't match the robot-centric axes we're used to
        rx = math.cos(alpha)*rho
        ry = -math.sin(alpha)*rho

        t = pose_theta + np.pi/2.
        # Convert detection from robot coordinates into world coordinates
        wx =  math.cos(t)*rx - math.sin(t)*ry + pose_x
        wy =  math.sin(t)*rx + math.cos(t)*ry + pose_y

        ################ ^ [End] Do not modify ^ ##################

        #print("Rho: %f Alpha: %f rx: %f ry: %f wx: %f wy: %f" % (rho,alpha,rx,ry,wx,wy))
        if wx >= 13:
            wx = 12.999
        if wy <= -10:
            wy = -9.999
        if rho < LIDAR_SENSOR_MAX_RANGE:
            # Part 1.3: visualize map gray values.
            # You will eventually REPLACE the following lines with a more robust version of the map
            # with a grayscale drawing containing more levels than just 0 and 1.
            map_x = 360-int(abs(wx*27.6923076923))
            map_y = int(abs(wy*30))
            map[map_y, map_x]+=5e-3
            # color = (g*256**2 + g*256+g)*255+=5e-3
            # display.setColor(color)
            # display.drawPixel()
            # g=360-abs(int(wx*30)),abs(int(wy*30))
            # color= (g*256**2+g*256+g)*255
            # map[wx,wy]=color

    # Draw the robot's current pose on the 360x360 display
    display.setColor(int(0xFF0000))
    display.drawPixel(360-abs(int(pose_x*30)), abs(int(pose_y*30)))

    ###################
    #
    # Controller
    #
    ###################
    if mode == 'manual':
        key = keyboard.getKey()
        while(keyboard.getKey() != -1): pass
        if key == keyboard.LEFT :
            vL = -MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.RIGHT:
            vL = MAX_SPEED
            vR = -MAX_SPEED
        elif key == keyboard.UP:
            vL = MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.DOWN:
            vL = -MAX_SPEED
            vR = -MAX_SPEED
        elif key == ord(' '):
            vL = 0
            vR = 0
        elif key == ord('S'):
            # Part 1.4: Filter map and save to filesystem
            map = map > 0.5  
            np.multiply(map, 1) 
            np.save("map.npy", map)
            print(map)
            map_display=(map*256**2+map*256+map)*255
            plt.imshow(map_display,origin='upper')
            plt.savefig("map.png")
            print("Map file saved")
        elif key == ord('L'):
            # You will not use this portion in Part 1 but here's an example for loading saved a numpy array
            map = np.load("map.npy")
            print("Map loaded")
        else: # slow down
            vL *= 0.75
            vR *= 0.75
    else: # not manual mode
        # Part 3.2: Feedback controller
        #STEP 1: Calculate the error
        rho = 0
        alpha = 0

        #STEP 2: Controller
        dX = 0
        dTheta = 0

        #STEP 3: Compute wheelspeeds
        vL = 0
        vR = 0

        # Normalize wheelspeed
        # (Keep the wheel speeds a bit less than the actual platform MAX_SPEED to minimize jerk)


    # Odometry code. Don't change vL or vR speeds after this line.
    # We are using GPS and compass for this lab to get a better pose but this is how you'll do the odometry
    pose_x += (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.cos(pose_theta)
    pose_y -= (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.sin(pose_theta)
    pose_theta += (vR-vL)/AXLE_LENGTH/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0

    # print("X: %f Z: %f Theta: %f" % (pose_x, pose_y, pose_theta))

    # Actuator commands
    robot_parts[MOTOR_LEFT].setVelocity(vL)
    robot_parts[MOTOR_RIGHT].setVelocity(vR)
    
while robot.step(timestep) != -1:
    # there is a bug where webots have to be restarted if the controller exits on Windows
    # this is to keep the controller running
    pass
