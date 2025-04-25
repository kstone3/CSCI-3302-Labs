"""lab5 controller."""
from os import path
from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard
import math
import numpy as np
from matplotlib import pyplot as plt
import heapq
import time
from scipy.signal import convolve2d # Uncomment if you want to use something else for finding the configuration space
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import ikpy.utils.plot as plot_utils
#from lab5_joint import getTargetFromObject, calculateIk, checkArmAtPosition, moveArmToTarget, reachArm, closeGrip, openGrip
#import lab5_joint
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

base_elements=["base_link", "base_link_Torso_joint", "Torso", "torso_lift_joint", "torso_lift_link", "torso_lift_link_TIAGo front arm_11367_joint", "TIAGo front arm_11367"]
my_chain = Chain.from_urdf_file("tiago_urdf.urdf", base_elements=["base_link", "base_link_Torso_joint", "Torso", "torso_lift_joint", "torso_lift_link", "torso_lift_link_TIAGo front arm_11367_joint", "TIAGo front arm_11367"])

#print(my_chain.links)
for link_id in range(len(my_chain.links)):

    # This is the actual link object
    link = my_chain.links[link_id]
    
    # I've disabled "torso_lift_joint" manually as it can cause
    # the TIAGO to become unstable.
    if link.name not in part_names or  link.name =="torso_lift_joint":
        print("Disabling {}".format(link.name))
        my_chain.active_links_mask[link_id] = False
        
# Initialize the arm motors and encoders.
motors = []
for link in my_chain.links:
    if link.name in part_names and link.name != "torso_lift_joint":
        motor = robot.getDevice(link.name)

        # Make sure to account for any motors that
        # require a different maximum velocity!
        if link.name == "torso_lift_joint":
            motor.setVelocity(0.07)
        else:
            motor.setVelocity(1)
            
        position_sensor = motor.getPositionSensor()
        position_sensor.enable(timestep)
        motors.append(motor)

for i in range(N_PARTS):
    robot_parts.append(robot.getDevice(part_names[i]))
    if i<10:
        robot_parts[i].setPosition(float(target_pos[i]))
        robot_parts[i].setVelocity(robot_parts[i].getMaxVelocity() / 2.0)
    else:
        robot_parts[i].setPosition(float(target_pos[i]))
        robot_parts[i].setVelocity(0)
    robot.step(timestep)
    
for _ in range (100):
    robot.step(timestep)

# The Tiago robot has a couple more sensors than the e-Puck
# Some of them are mentioned below. We will use its LiDAR for Lab 5

range_finder = robot.getDevice("RangeFinder")
range_finder.enable(timestep)
camera = robot.getDevice("camera")
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
#mode = 'manual' # Part 1.1: manual mode
#mode = 'planner'
# mode = 'autonomous'
mode = 'picknplace'


fix_map=False
###################
#
# Planner
#
###################
if mode == 'planner':
    print("Planning route ...")
    # Part 2.3: Provide start and end in world coordinate frame and convert it to map's frame
    start_w = (-7.97,-4.84) # (Pose_X, Pose_Y) in meters
    #start_w=(-7.1,-5.3)
    #test1: start_w = (-3, -3.3)
    #test2: start_w = (-8.5, -2)
    # while np.any(np.isnan(gps.getValues())):
    #     robot.step(timestep)
    #     print("Waiting for GPS signal")
    # start_w=(gps.getValues()[0]-1,gps.getValues()[1])
    # end_w = (10,7) # (Pose_X, Pose_Y) in meters
    end_w = (10,7)

    def world_to_map(world_x, world_y):
        # map_x = int((world_x + 12) * 30)
        # map_y = int(-world_y * 30)
        map_x = 360-int(abs(world_x*30))
        map_y = int(abs(world_y*30))
        return map_x, map_y

    # Convert the start_w and end_w from the webots coordinate frame into the map frame
    start = world_to_map(start_w[0], start_w[1]) # (x, y) in 360x360 map
    end = world_to_map(end_w[0], end_w[1]) # (x, y) in 360x360 map

    # Part 2.3: Implement A* or Dijkstra's Algorithm to find a path
    def path_planner(path_planner_map, start_planner, end_planner):
        '''
        :param map: A 2D numpy array of size 360x360 representing the world's cspace with 0 as free space and 1 as obstacle
        :param start: A tuple of indices representing the start cell in the map
        :param end: A tuple of indices representing the end cell in the map
        :return: A list of tuples as a path from the given start to the given end in the given maze
        '''
        
        if path_planner_map[start_planner[0], start_planner[1]] != 0:
            print("Start is not traversable")
            return []
        if path_planner_map[end_planner[0], end_planner[1]] != 0:
            print("End is not traversable")
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
                    if 0 <= nx < path_planner_map.shape[0] and 0 <= ny < path_planner_map.shape[1] and path_planner_map[nx, ny] == 0:
                        neighbors.append((nx, ny))
            return neighbors

        def move_cost(current, neighbor):
            return math.sqrt(2) if current[0] != neighbor[0] and current[1] != neighbor[1] else 1

        open = []
        heapq.heappush(open, (heuristic(start_planner, end_planner), 0, start_planner))
        prev_location = {}
        current_cost = {start_planner: 0}
        
        while open:
            current_priority, cost, current = heapq.heappop(open)
            if current == end:
                path = []
                while current != start_planner:
                    path.append(current)
                    current = prev_location[current]
                path.append(start_planner)
                path.reverse()
                return path
            
            for neighbor in get_neighbors(current):
                new_cost=current_cost[current] + move_cost(current, neighbor)
                if neighbor not in current_cost or new_cost<current_cost[neighbor]:
                    current_cost[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, end_planner)
                    heapq.heappush(open, (priority, new_cost, neighbor))
                    prev_location[neighbor] = current
        
        return [end_planner]
    
    # Part 2.1: Load map (map.npy) from disk and visualize it
    map = np.load("map.npy")
    if fix_map:
        map = map.T
        np.save("map.npy", map)
    plt.imshow(map.T)
    #plt.show()
    plt.savefig("map.png")
    
    # Part 2.2: Compute an approximation of the “configuration space”
    configured_map = np.zeros(map.shape)
    obstacle_detected = np.argwhere(map>0)
    footprint_radius = 8
    for x, y in obstacle_detected:
        x_min, x_max = max(0, x - footprint_radius), min(map.shape[0], x + footprint_radius)
        y_min, y_max = max(0, y - footprint_radius), min(map.shape[1], y + footprint_radius)
        configured_map[x_min:x_max, y_min:y_max] = 1
    if not np.all(np.isin(configured_map, [0, 1])): print("Error in configured map: contains values other than 0 and 1")
    plt.imshow(configured_map.T)
    # plt.show()
    np.save("configured_map.npy", configured_map)
    plt.savefig("Configured_map.png")
    # Part 2.3 continuation: Call path_planner
    # start_world_coords = (pose_x, pose_y)
    # end_world_coords = (pose_x + 1, pose_y + 1) # Replace with actual end coordinates
    path = path_planner(configured_map, start, end)
    # Part 2.4: Turn paths into waypoints and save on disk as path.npy and visualize it
    plt.scatter(path[0][0], path[0][1], c='green', marker='o')
    plt.scatter(path[-1][0], path[-1][1], c='blue', marker='x')
    if path:
        path_x, path_y = zip(*path)
        # plt.scatter(path_x, path_y, c='black', marker='s')
        plt.plot(path_x,path_y, c='red', linewidth=2)
    plt.savefig("path.png")
    def convert_to_world(path):
        return [((x - 360) / 30, y / -30) for x, y in path]
    
    waypoints = []

    if path:
        waypoints = convert_to_world(path)
        np.save("path.npy", np.array(waypoints))
        print("Path found")
    print("Done planning route")
    mode = 'autonomous'
######################
#
# Map Initialization
#
######################

# Part 1.2: Map Initialization

# Initialize your map data structure here as a 2D floating point array
map = np.zeros(shape=[360,360])
waypoints = []

if mode == 'autonomous':
    # Part 3.1: Load path from disk and visualize it
    waypoints = np.load("path.npy") # Replace with code to load your path
    print(waypoints.size)
    index=0
    # configured_map=np.load("configured_map.npy")
    # plt.imshow(configured_map.T, origin="upper")
    # plt.savefig("configured_map.png")
    # Plot start and end points
    # plt.scatter(waypoints[0][0], waypoints[0][1], c='green', marker='o')
    # plt.scatter(waypoints[-1][0], waypoints[-1][1], c='blue', marker='x')
    # # Plot path if found
    # if waypoints.size > 0:
    #     path_x, path_y = zip(*waypoints)
    #     # plt.scatter(path_x, path_y, c='black', marker='s')
    #     plt.plot(path_x,path_y, c='red', linewidth=2)
    # plt.savefig("path.png")
state = 0 # use this to iterate through your path


def getTargetFromObject(recognized_objects):
    ''' Gets a target vector from a list of recognized objects '''

    # Get the first valid target
    target = recognized_objects[0].getPosition()

    # Convert camera coordinates to IK/Robot coordinates
    # offset_target = [-(target[2])+0.22, -target[0]+0.08, (target[1])+0.97+0.2]
    offset_target = [-(target[2])+0.22, -target[0]+0.06, (target[1])+0.97+0.2]

    return offset_target

def lookForTarget(recognized_objects):
    target_item = "orange"
    if len(recognized_objects) > 0:

        for item in recognized_objects:
            if target_item in str(item.getModel()):

                target = recognized_objects[0].getPosition()
                dist = abs(target[2])

                if dist < 5:
                    return True
                
def checkArmAtPosition(ikResults, cutoff=0.00005):
    '''Checks if arm at position, given ikResults'''
    
    # Get the initial position of the motors
    initial_position = [0,0,0,0] + [m.getPositionSensor().getValue() for m in motors] + [0,0,0,0]

    # Calculate the arm
    arm_error = 0
    for item in range(14):
        arm_error += (initial_position[item] - ikResults[item])**2
    arm_error = math.sqrt(arm_error)

    if arm_error < cutoff:
        if vrb:
            print("Arm at position.")
        return True
    return False

def calculateIk(offset_target,  orient=True, orientation_mode="Y", target_orientation=[0,0,1]):
    '''
    This will calculate the iK given a target in robot coords
    Parameters
    ----------
    param offset_target: a vector specifying the target position of the end effector
    param orient: whether or not to orient, default True
    param orientation_mode: either "X", "Y", or "Z", default "Y"
    param target_orientation: the target orientation vector, default [0,0,1]

    Returns
    ----------
    rtype: bool
        returns: whether or not the arm is at the target
    '''
    # # Get the number of links in the chain
    # num_links = len(my_chain.links)

    # # Create initial position array with the correct size
    # initial_position = [0] * num_links
    # #initial_position = [0, 0, 0, 0] + [m.getPositionSensor().getValue() for m in motors] +[0, 0, 0]

    # # Map each motor to its corresponding link position
    # motor_idx = 0
    # for i in range(num_links):
    #     link_name = my_chain.links[i].name
    #     if link_name in part_names and link_name != "torso_lift_joint":
    #         if motor_idx < len(motors):
    #             initial_position[i] = motors[motor_idx].getPositionSensor().getValue()
    #             motor_idx += 1

    # # Calculate IK
    # ikResults = my_chain.inverse_kinematics(
    #     offset_target, 
    #     initial_position=initial_position,
    #     target_orientation=target_orientation, 
    #     orientation_mode=orientation_mode
    # )

    # # Validate result
    # position = my_chain.forward_kinematics(ikResults)
    # squared_distance = math.sqrt(
    #     (position[0, 3] - offset_target[0])**2 + 
    #     (position[1, 3] - offset_target[1])**2 + 
    #     (position[2, 3] - offset_target[2])**2
    # )
    # print(f"IK calculated with error - {squared_distance}")

    # return ikResults

    #Get the initial position of the motors
    def bound_joint_values(initial_pos, bounds):
        return [max(min(val, ub), lb) for val, (lb, ub) in zip(initial_pos, bounds)]
    def get_bounded_initial_position():
        initial_position = [0, 0, 0, 0] + [m.getPositionSensor().getValue() for m in motors] +[0, 0, 0]
        bounds = [
            (-float('inf'), float('inf')),  # 0
            (-float('inf'), float('inf')),  # 1
            (0.0, 0.35),                    # 2
            (-float('inf'), float('inf')),  # 3
            (0.07, 2.68),                   # 4 
            (-1.5, 1.02),                   # 5 
            (-3.46, 1.5),                   # 6
            (-0.32, 2.29),                  # 7 
            (-2.07, 2.07),                  # 8 
            (-1.39, 1.39),                  # 9 
            (-2.07, 2.07),                  # 10 
            (-float('inf'), float('inf')),  # 11
            (-float('inf'), float('inf')),  # 12
            (0.0, 0.045),                   # 13
        ]

        return bound_joint_values(initial_position, bounds)
    
    initial_position = get_bounded_initial_position()
    #initial_position = [0, 0, 0, 0] + [m.getPositionSensor().getValue() for m in motors] +[0, 0, 0] #
    print(initial_position)

    #Calculate IK
    ikResults = my_chain.inverse_kinematics(offset_target, initial_position=initial_position,  target_orientation = [0,0,1], orientation_mode="Y")

    # Use FK to calculate squared_distance error
    position = my_chain.forward_kinematics(ikResults)

    # This is not currently used other than as a debug measure...
    squared_distance = math.sqrt((position[0, 3] - offset_target[0])**2 + (position[1, 3] - offset_target[1])**2 + (position[2, 3] - offset_target[2])**2)
    print("IK calculated with error - {}".format(squared_distance))

    # Reset the ikTarget (deprec)
    # ikTarget = offset_target
    
    return ikResults
    
    # Legacy code for visualizing
        # import matplotlib.pyplot
        # from mpl_toolkits.mplot3d import Axes3D
        # ax = matplotlib.pyplot.figure().add_subplot(111, projection='3d')

        # my_chain.plot(ikResults, ax, target=ikTarget)
        # matplotlib.pyplot.show()
vrb = True      
def moveArmToTarget(ikResults):
    '''Moves arm given ikResults'''
    # Set the robot motors
    for res in range(len(ikResults)):
        if my_chain.links[res].name in part_names:
            # This code was used to wait for the trunk, but now unnecessary.
            # if abs(initial_position[2]-ikResults[2]) < 0.1 or res == 2:
            robot.getDevice(my_chain.links[res].name).setPosition(ikResults[res])
            if vrb:
                print("Setting {} to {}".format(my_chain.links[res].name, ikResults[res]))

def closeGrip():
    robot.getDevice("gripper_right_finger_joint").setPosition(0.0)
    robot.getDevice("gripper_left_finger_joint").setPosition(0.0)

    # r_error = abs(robot.getDevice("gripper_right_finger_joint").getPositionSensor().getValue() - 0.01)
    # l_error = abs(robot.getDevice("gripper_left_finger_joint").getPositionSensor().getValue() - 0.01)
    
    # print("ERRORS")
    # print(r_error)
    # print(l_error)

    # if r_error+l_error > 0.0001:
    #     return False
    # else:
    #     return True


if mode == 'picknplace':
    # Part 4: Use the function calls from lab5_joints using the comments provided there
    ## use path_planning to generate paths
    ## do not change start_ws and end_ws below
    start_ws = [(3.7, 5.7)]
    end_ws = [(10.0, 9.3)]
    # start = world_to_map(start_ws[0], start_ws[1]) # (x, y) in 360x360 map
    # end = world_to_map(end_ws[0], end_ws[1]) # (x, y) in 360x360 map
    target = "orange"
    while robot.step(timestep) != -1:
        recognized_obj = camera.getRecognitionObjects()
        # print("recognized:", recognized_obj)
        if lookForTarget(recognized_obj) == True: 
            target_pos = getTargetFromObject(recognized_obj)
            # print("target_pos:", target_pos)
            Ikresult = calculateIk(target_pos)
            print("IK: ", Ikresult)
            moveArmToTarget(Ikresult)
            checkArmAtPosition(Ikresult)
            #closeGrip()
            
            
           
    pass
robot_semi_state = 0
run_start=True
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
    # rad = -((math.atan2(n[0], n[2]))-1.5708)
    rad = -((math.atan2(n[0], n[2]))-np.pi)
    pose_theta = rad

    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]

    # vL=0
    # vR=0

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
        if wx >= 12:
            wx = 11.999
        if wy <= -12:
            wy = -11.999
        if rho < LIDAR_SENSOR_MAX_RANGE:
            # Part 1.3: visualize map gray values.
            # You will eventually REPLACE the following lines with a more robust version of the map
            # with a grayscale drawing containing more levels than just 0 and 1.
            map_x = 360-int(abs(wx*30))
            map_y = int(abs(wy*30))
            map[map_x,map_y]+=5e-3
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
            plt.imshow(map_display.T,origin='upper')
            plt.savefig("map.png")
            print("Map file saved")
        elif key == ord('L'):
            # You will not use this portion in Part 1 but here's an example for loading saved a numpy array
            map = np.load("map.npy")
            map_display=(map*256**2+map*256+map)*255
            plt.imshow(map_display.T,origin='upper')
            plt.savefig("map.png")
            print("Map loaded")
        else: # slow down
            vL *= 0.75
            vR *= 0.75
    elif mode=='autonomous': # not manual mode
        # if run_start and robot.time()>1: 
        #     time.sleep(5)
        #     run_start = False
        # Part 3.2: Feedback controller
        #STEP 1: Calculate the error
        x_goal, y_goal=waypoints[index]
        rho=np.sqrt((x_goal - pose_x) ** 2 + (y_goal - pose_y) ** 2)  # Distance to the goal
        theta_g = np.arctan2(y_goal - pose_y, x_goal - pose_x)  # Angle to the goal
        alpha = theta_g - pose_theta  # Difference from robot's heading
        if alpha<-np.pi:
            alpha += 2*np.pi
        if index < len(waypoints) - 1:
            x_next, y_next = waypoints[index + 1]
        else:
            print("Goal reached")
            vL=0
            vR=0
        # print("x_next:", x_next, "y_next: ", y_next)
        print("rho:", rho, "theta_g: ", theta_g, "alpha: ", alpha)
        print("x_goal: ", x_goal, "y_goal: ", y_goal)
        print("index: ", index)
        # # Normalize wheelspeed
        # # (Keep the wheel speeds a bit less than the actual platform MAX_SPEED to minimize jerk)
        # # vL=max(min(0.5*alpha+2*rho, MAX_SPEED-3),-MAX_SPEED+4)
        # # vR=max(min(-0.5*alpha+2*rho, MAX_SPEED-3),-MAX_SPEED+4)
        # vL=max(min(2.5*alpha+6*rho, MAX_SPEED),-MAX_SPEED)
        # vR=max(min(-2.5*alpha+6*rho, MAX_SPEED),-MAX_SPEED)
        # theta_goal_orientation = np.arctan2(y_next - y_goal, x_next - x_goal)
        # eta = theta_goal_orientation - pose_theta
        # eta = (eta + np.pi) % (2 * np.pi) - np.pi
        # Normalize wheelspeed
        # (Keep the wheel speeds a bit less than the actual platform MAX_SPEED to minimize jerk)
        # vL=max(min(0.5*alpha+2*rho, MAX_SPEED-3),-MAX_SPEED+4)
        # vR=max(min(-0.5*alpha+2*rho, MAX_SPEED-3),-MAX_SPEED+4)
        # vL=max(min(-2.5*alpha+6*rho, MAX_SPEED),-MAX_SPEED)
        # vR=max(min(2.5*alpha+6*rho, MAX_SPEED),-MAX_SPEED)
        vL=max(min(-12*alpha+12.56*rho, MAX_SPEED*0.5),-MAX_SPEED*0.5)
        vR=max(min(12*alpha+12.56*rho, MAX_SPEED*0.5),-MAX_SPEED*0.5)
        theta_goal_orientation = np.arctan2(y_next - y_goal, x_next - x_goal)
        eta = theta_goal_orientation - pose_theta
        eta = (eta + np.pi) % (2 * np.pi) - np.pi
        # if (abs(alpha) > 0.1) and (robot_semi_state == 0):
        #     if alpha > 0:
        #         vL = -MAX_SPEED / 4
        #         vR = MAX_SPEED / 4
        #     else:
        #         vL = MAX_SPEED / 4
        #         vR = -MAX_SPEED / 4
        # else:
        #     if robot_semi_state == 0:
        #         # Bearing error is small; proceed to drive forward.
        #         vL = 0
        #         vR = 0
        #         robot_semi_state = 1
        #     elif robot_semi_state == 1:
        #         # Drive forward while reducing position error (ρ).
        #         if (rho > 0.05):
        #             vL =  MAX_SPEED / 2
        #             vR =  MAX_SPEED / 2
        #         else:
        #             robot_semi_state = 2
        #     elif robot_semi_state == 2:
        #         # Rotate to adjust the heading error (η).
        #         if abs(eta) > 0.1:
        #             if eta > 0:
        #                 vL = -MAX_SPEED / 4
        #                 vR = MAX_SPEED / 4
        #             else:
        #                 vL = MAX_SPEED / 4
        #                 vR = -MAX_SPEED / 4
        #         else:
        #             # Finished the turn; stop and update the waypoint.
        #             if index < len(waypoints) - 1: index += 1
        #             else: 
        #                 print("Goal reached")
        #                 vL=0
        #                 vR=0
        #             robot_semi_state = 0
        print("vL:", vL, "vR: ", vR)
        # print("dx: ", dx, "dy: ", dy)
        # if rho < 0.1:
        #     state += 1
        #     if state >= len(waypoints):
        #         state = 0
        #         vL=0
        #         vR=0
        if rho<0.25: 
            if index < len(waypoints) - 1: index += 1
            else: 
                print("Goal reached")
                vL=0
                vR=0
        # vL = 0
        # vR = 0

    # Odometry code. Don't change vL or vR speeds after this line.
    # We are using GPS and compass for this lab to get a better pose but this is how you'll do the odometry
    # pose_x += (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.cos(pose_theta)
    # pose_y -= (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.sin(pose_theta)
    # pose_theta += (vR-vL)/AXLE_LENGTH/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0

    #print("X: %f Z: %f Theta: %f" % (pose_x, pose_y, pose_theta))

    # Actuator commands
    robot_parts[MOTOR_LEFT].setVelocity(vL)
    robot_parts[MOTOR_RIGHT].setVelocity(vR)
    
while robot.step(timestep) != -1:
    # there is a bug where webots have to be restarted if the controller exits on Windows
    # this is to keep the controller running
    pass
