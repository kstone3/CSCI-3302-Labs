"""grocery controller."""

# Apr 1, 2025

import time
from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard
import math
import numpy as np
from matplotlib import pyplot as plt
import heapq
import random
import scipy.ndimage
import drive_ik

#Initialization
print("=== Initializing Grocery Shopper...")
#Consts
MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
RADIUS = MAX_SPEED_MS/MAX_SPEED*4
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12
LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 5.5 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint",
              "gripper_left_finger_joint","gripper_right_finger_joint")

# 

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.35, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf',0.045,0.045)

robot_parts={}
for i, part_name in enumerate(part_names):
    robot_parts[part_name]=robot.getDevice(part_name)
    robot_parts[part_name].setPosition(float(target_pos[i]))
    robot_parts[part_name].setVelocity(robot_parts[part_name].getMaxVelocity() / 2.0)

# Enable gripper encoders (position sensors)
left_gripper_enc=robot.getDevice("gripper_left_finger_joint_sensor")
right_gripper_enc=robot.getDevice("gripper_right_finger_joint_sensor")
left_gripper_enc.enable(timestep)
right_gripper_enc.enable(timestep)

# Enable Camera
camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)

#Enable keyboard
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# Enable GPS and compass localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# Enable LiDAR
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# Enable display
display = robot.getDevice("display")

def a_star(path_planner_map, start_planner, end_planner):
    '''
    :param path_planner_map: A 2D numpy array of size 360x360 representing the world's cspace with 0 as free space and 1 as obstacle
    :param start_planner: A tuple of indices representing the start cell in the map
    :param end_planner: A tuple of indices representing the end cell in the map
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
        if current == end_planner:  # Changed from 'end' to 'end_planner'
            path = []
            while current != start_planner:
                path.append(current)
                current = prev_location[current]
            path.append(start_planner)
            path.reverse()
            return path
        
        for neighbor in get_neighbors(current):
            new_cost = current_cost[current] + move_cost(current, neighbor)
            if neighbor not in current_cost or new_cost < current_cost[neighbor]:
                current_cost[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, end_planner)
                heapq.heappush(open, (priority, new_cost, neighbor))
                prev_location[neighbor] = current
    return [end_planner]



class Node:
    """
    Node for RRT/RRT* Algorithm. This is what you'll make your graph with!
    """
    def __init__(self, pt, parent=None):
        self.point = pt # n-Dimensional point
        self.parent = parent # Parent node
        self.path_from_parent = [] # List of points along the way from the parent node (for visualization)
        
def get_random_valid_vertex(state_valid, bounds, obstacles):
    vertex = None
    while vertex is None: # Get starting vertex
        pt = np.random.rand(bounds.shape[0]) * (bounds[:,1]-bounds[:,0]) + bounds[:,0]
        if state_valid(pt):
            vertex = pt
    return vertex

def get_nearest_vertex(node_list, q_point):
    '''
    @param node_list: List of Node objects
    @param q_point: n-dimensional array representing a point
    @return Node in node_list with closest node.point to query q_point
    '''
    # TODO: Your Code Here
    nearest_vertex=node_list[0]
    min_dist=np.linalg.norm(q_point-nearest_vertex.point)
    for node in node_list:
        dist=np.linalg.norm(q_point-node.point)
        if dist<min_dist:
            nearest_vertex=node
            min_dist=dist
    return nearest_vertex

def steer(from_point, to_point, delta_q):
    '''
    @param from_point: n-Dimensional array (point) where the path to "to_point" is originating from (e.g., [1.,2.])
    @param to_point: n-Dimensional array (point) indicating destination (e.g., [0., 0.])
    @param delta_q: Max path-length to cover, possibly resulting in changes to "to_point" (e.g., 0.2)
    @returns path: list of points leading from "from_point" to "to_point" (inclusive of endpoints)  (e.g., [ [1.,2.], [1., 1.], [0., 0.] ])
    '''

    path = []

    # TODO: Figure out if you can use "to_point" as-is, or if you need to move it so that it's only delta_q distance away
    from_pt=np.array(from_point)
    to_pt=np.array(to_point)
    dir=to_pt-from_pt
    dist=np.linalg.norm(dir)
    if dist>delta_q: to_pt=from_pt+(dir/dist)*delta_q
    path=np.linspace(from_pt,to_pt,num=10)
    path[0]=from_pt
    path[-1]=to_pt
    return path.tolist()

def rrt_star(state_bounds, obstacles, state_is_valid, starting_point, goal_point, k, delta_q):
    '''
    TODO: Implement the RRT* algorithm here, making use of the provided state_is_valid function

    @param state_bounds: matrix of min/max values for each dimension (e.g., [[0,1],[0,1]] for a 2D 1m by 1m square)
    @param state_is_valid: function that maps states (N-dimensional Real vectors) to a Boolean (indicating free vs. forbidden space)
    @param k: Number of points to sample
    @param delta_q: Maximum distance allowed between vertices
    @returns List of RRT* graph nodes
    '''

    node_list = []
    start_node=Node(starting_point,parent=None)
    start_node.cost=0.0
    node_list.append(start_node)
    neighbor_radius=0.2  
    goal_bias=0.10
    for i in range(k):
        if goal_point is not None and random.random()<goal_bias: q_rand=goal_point
        else: q_rand=get_random_valid_vertex(state_is_valid,state_bounds,obstacles)
        nearest_node = get_nearest_vertex(node_list,q_rand)
        if not hasattr(nearest_node,'cost'): nearest_node.cost=0
        new_path=steer(nearest_node.point,q_rand,delta_q)
        q_new=np.array(new_path[-1])
        valid=True
        for pt in new_path:
            if not state_is_valid(np.array(pt)):
                valid = False
                break
        if not valid: continue
        new_node=Node(q_new,parent=nearest_node)
        new_node.path_from_parent=new_path
        new_node.cost=nearest_node.cost+np.linalg.norm(q_new-nearest_node.point)
        node_list.append(new_node)
        neighbors=[]
        for node in node_list:
            dist=np.linalg.norm(node.point-new_node.point)
            if dist<=neighbor_radius:
                path_test=steer(node.point,new_node.point,delta_q)
                path_valid=True
                for p in path_test:
                    if not state_is_valid(np.array(p)):
                        path_valid = False
                        break
                if path_valid:
                    if not hasattr(node,'cost'): node.cost=0
                    neighbors.append(node)
        best_parent=new_node.parent
        best_cost=new_node.cost
        best_path=new_node.path_from_parent
        for nbr in neighbors:
            path_test=steer(nbr.point,new_node.point,delta_q)
            path_valid=True
            for p in path_test:
                if not state_is_valid(np.array(p)):
                    path_valid=False
                    break
            if not path_valid: continue
            cost_through_nbr=nbr.cost+np.linalg.norm(new_node.point-nbr.point)
            if cost_through_nbr<best_cost:
                best_parent=nbr
                best_cost=cost_through_nbr
                best_path=path_test
        new_node.parent=best_parent
        new_node.path_from_parent=best_path
        new_node.cost=best_cost
        for nbr in neighbors:
            if nbr==new_node.parent: continue
            path_test=steer(new_node.point,nbr.point,delta_q)
            path_valid=True
            for p in path_test:
                if not state_is_valid(np.array(p)):
                    path_valid=False
                    break
            if not path_valid: continue
            cost_through_new=new_node.cost+np.linalg.norm(nbr.point-new_node.point)
            if cost_through_new<nbr.cost:
                nbr.parent=new_node
                nbr.path_from_parent=path_test
                nbr.cost=cost_through_new
        if goal_point is not None and np.linalg.norm(new_node.point-np.array(goal_point))<1e-5:
            print("Goal reached (RRT*)")
            return node_list
    return node_list

def compute_control(current_pose, waypoint):
    # current_pose: (pose_x, pose_y, pose_theta)
    # waypoint: np.array([wx, wy])
    dx = waypoint[0] - current_pose[0]
    dy = waypoint[1] - current_pose[1]
    desired_angle = math.atan2(dy, dx)
    angle_error = desired_angle - current_pose[2]
    # Normalize the angle to [-pi, pi]
    angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
    # Distance to the waypoint
    distance = math.hypot(dx, dy)
    return distance, angle_error

def grid_to_world(cell, scale_x, scale_y):
    row, col = cell
    wx = col / scale_x - 14
    wy = 7 - row / scale_y
    return np.array([wx, wy])

def choose_frontier(robot_cell, frontiers):
    if not frontiers:
        return None
    r_robot, c_robot = robot_cell
    min_dist = float('inf')
    best = None
    for cell in frontiers:
        dr = cell[0] - r_robot
        dc = cell[1] - c_robot
        dist = math.hypot(dr, dc)
        if dist < min_dist:
            min_dist = dist
            best = cell
    return best

def detect_frontiers(binary_map):
    # Create a 3x3 kernel that sums neighbor values.
    kernel = np.ones((3, 3))
    # Convolve the binary map to get the sum of neighbors
    neighbor_sum = scipy.ndimage.convolve(binary_map, kernel, mode='constant', cval=0)
    # Define frontiers as free/unknown cells that have at least one neighbor that is also free/unknown.
    frontiers = list(zip(*np.where((binary_map == 0) & (neighbor_sum < 9))))
    return frontiers



# Odometry
pose_x     = 0
pose_y     = 0
pose_theta = 0

vL = 0
vR = 0

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis

map = None
map = np.zeros(shape=[360,360])
state="manual_map"
current_path_world = []  # List of waypoints (in world coords)
current_waypoint_index = 0
# Define the planning update frequency (e.g., every 50 timesteps)
planning_interval = 100  
planning_counter = 100

# ------------------------------------------------------------------
# Helper Functions


gripper_status="closed"
print("=== Running Grocery Shopper...")
# Main Loop
while robot.step(timestep) != -1:
    
    pose_y = -gps.getValues()[1]
    pose_x = -gps.getValues()[0]

    print(f'x: {-gps.getValues()[0]} y: {-gps.getValues()[1]}')

    n = compass.getValues()
    rad = ((math.atan2(n[0], n[1])))
    pose_theta = rad
    
    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]
    
    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]

        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue

        # The Webots coordinate system doesn't match the robot-centric axes we're used to
        rx = -math.cos(alpha)*rho
        ry = math.sin(alpha)*rho


        # Convert detection from robot coordinates into world coordinates
        wx =  math.cos(pose_theta)*rx - math.sin(pose_theta)*ry + pose_x
        wy =  +(math.sin(pose_theta)*rx + math.cos(pose_theta)*ry) + pose_y


    
        ################ ^ [End] Do not modify ^ ##################

        # print("Rho: %f Alpha: %f rx: %f ry: %f wx: %f wy: %f" % (rho,alpha,rx,ry,wx,wy))

        if rho < LIDAR_SENSOR_MAX_RANGE:
            # Part 1.3: visualize map gray values.

            # You will eventually REPLACE the following 3 lines with a more robust version of the map
            # with a grayscale drawing containing more levels than just 0 and 1.
            scale_x = 360 / 30.0   # ~12.857 pixels per meter for x
            scale_y = 360 / 16.0   # ~25.714 pixels per meter for y
            
            # Map world coordinates to grid indices
            x = int((wx + 15) * scale_x)
            y = int((8-wy) * scale_y)
            if x >= 360 or y >=360 or x < 0 or y <0:
                print("outta range")
            else: 
                if map[x][y] >= 1:
                    pass
                else:    
                    map[x][y]+= 0.05 
                    # g = map[x][y]
                    # color = int(g*256**2+g*256+g)*255 
                    # if color > 0xffffff:
                    #     color = int(0xffffff)
                    # display.setColor(color)
                    # display.drawPixel(x,y)
                    # if map[x][y] >= .9:
                    #     # np.save('map.npy', map)
                    #     print("saved")
    

    
    if(gripper_status=="open"):
        # Close gripper, note that this takes multiple time steps...
        robot_parts["gripper_left_finger_joint"].setPosition(0)
        robot_parts["gripper_right_finger_joint"].setPosition(0)
        if right_gripper_enc.getValue()<=0.005:
            gripper_status="closed"
    else:
        # Open gripper
        robot_parts["gripper_left_finger_joint"].setPosition(0.045)
        robot_parts["gripper_right_finger_joint"].setPosition(0.045)
        if left_gripper_enc.getValue()>=0.044:
            gripper_status="open"
    
    if state=="manual_map":
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
            r_robot = int((8 - pose_y) * scale_y)
            c_robot = int((pose_x + 15) * scale_x)
            map = map > 0.5  
            np.multiply(map, 1) 
            np.save("map.npy", map)
            # print(map)
            map_display=(map*256**2+map*256+map)*255
            plt.scatter(r_robot,c_robot,marker='x',color='red')
            plt.imshow(map_display,origin='upper')
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
            
    if state == "auto_map":
        # Increment the planning counter each simulation step
        planning_counter += 1

        # Convert your occupancy grid to a binary occupancy map
        binary_map = (map > 0.5).astype(np.uint8)
        # Compute the robot's current grid cell using your scale factors:
        r_robot = int((8 - pose_y) * scale_y)
        c_robot = int((pose_x + 15) * scale_x)
        print("RX: ", r_robot, "CY: ", c_robot)
        robot_cell = (r_robot, c_robot)

        # Only perform expensive frontier detection and A* planning at the defined interval.
        if planning_counter >= planning_interval:
            np.save("map.npy", binary_map)
            # print(map)
            map_display=(binary_map*256**2+binary_map*256+binary_map)*255
            plt.imshow(map_display,origin='upper')
            plt.scatter(r_robot,c_robot,marker='x',color='red')
            plt.savefig("map.png")
            print("Map file saved")
            planning_counter = 0  # Reset the counter after planning

            frontiers = detect_frontiers(binary_map)
            goal_cell = choose_frontier(robot_cell, frontiers)
            if goal_cell is not None:
                path_grid = a_star(binary_map, robot_cell, goal_cell)
                # Convert the grid path to world coordinates:
                current_path_world = [grid_to_world(cell, scale_x, scale_y) for cell in path_grid]
                current_waypoint_index = 0
            else:
                # No frontier found; exploration complete.
                vL = 0
                vR = 0
                print("Exploration complete!")
                break  # Optionally stop the controller

        # Regardless of planning (which only updates every planning_interval steps),
        # the robot continues to follow the most recent plan.
        if current_path_world:
            # Check whether we've exhausted the path
            if current_waypoint_index >= len(current_path_world):
                print("Reached the end of the current path. Replanning...")
                current_path_world = []   # Clear the current path to trigger a replan next cycle.
                current_waypoint_index = 0
            else:
                current_waypoint = current_path_world[current_waypoint_index]
                distance, angle_error = compute_control((pose_x, pose_y, pose_theta), current_waypoint)
                # Controller gains; adjust these as needed:
                K_linear = 1.0
                K_angular = 2.0
                if abs(angle_error) > 0.1:
                    vL = -K_angular * angle_error
                    vR =  K_angular * angle_error
                else:
                    vL = K_linear * distance - K_angular * angle_error
                    vR = K_linear * distance + K_angular * angle_error
                # Proceed to next waypoint if close enough:
                if distance < 0.1:
                    current_waypoint_index += 1
    #     print("MOVE LIDAR UP to z = 0.5 !!!!!!!!!")
    #     # robot_parts[12].setPosition(0)
    #     # robot_parts[13].setPosition(0)
    #     time.sleep(2)

    #     path = np.load("astar_path.npy")
    #     drive_obj = drive_ik.Drive_IK_Controller(gps, compass, AXLE_LENGTH, RADIUS, MAX_SPEED)
    #     drive_obj.start_path(path,pose_x,pose_y,pose_theta)
    #     while robot.step(timestep) != -1:
    #         l,r = drive_obj.path_step(pose_x,pose_y,pose_theta)
    #         if (l,r) == (None,None):
    #             break
            
    #         # map = mapping.map_current_view(map, translation.get_current_position(gps,compass), lidar, lidar_offsets, LIDAR_SENSOR_MAX_RANGE, translation.local_size, save_file="map.npy")
    #         # plt.imsave("map.png", map)

    #         robot_parts[MOTOR_LEFT].setVelocity(l)
    #         robot_parts[MOTOR_RIGHT].setVelocity(r)
    #     plt.imsave("map.png", map)    
    #     robot_parts[MOTOR_LEFT].setVelocity(0)
    #     robot_parts[MOTOR_RIGHT].setVelocity(0) 
    #     # If we have reached our current waypoint or there is no valid path
    #     # if not current_path_world or current_waypoint_index >= len(current_path_world):
    #     #     planning_counter = planning_interval  # Force a replan on the next cycle.
    # elif state == "path_processing":
    #     robot_parts[MOTOR_LEFT].setVelocity(0)
    #     robot_parts[MOTOR_RIGHT].setVelocity(0) 

    #     print("generating path")
    #     points = [(395,160),(395,120),(40,120),(40,50),(430,50),(430,280),(40,280),(40,200),(440,200)]
    #     new_path = generate_complex_path("collection_map.npy", points, save_path_fname="astar_path.npy", save_img_fname="vis_with_path.png")
    #     print("path generated")
    #     break


                
    robot_parts["wheel_left_joint"].setVelocity(vL/5)
    robot_parts["wheel_right_joint"].setVelocity(vR/5)
    print("pose_x: ", pose_x, "pose_y: ", pose_y, "pose_theta: ", pose_theta)
    print("vL: ", vL, "vR: ", vR)