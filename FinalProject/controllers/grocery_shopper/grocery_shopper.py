"""grocery controller."""

# Apr 1, 2025

from collections import deque
from os import path
from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard
import math
import numpy as np
from matplotlib import pyplot as plt
import heapq
import random
import cv2
from matplotlib.colors import ListedColormap

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

def grid_to_world(cell, scale_x, scale_y):
    col,row = cell
    wx = col/scale_x - x_dim/2
    wy = row/scale_y - y_dim/2
    return np.array([wx, wy])

def bresenham(x0, y0, x1, y1):
    """
    Bresenham’s line algorithm between (x0,y0) and (x1,y1).
    Returns list of all integer (x,y) on the line, including both endpoints.
    """
    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy
    x, y = x0, y0

    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x   += sx
        if e2 <  dx:
            err += dx
            y   += sy

    return cells

# Odometry
pose_x     = 0
pose_y     = 0
pose_theta = 0

vL = 0
vR = 0

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis

# map = np.zeros(shape=[360,360])
map=np.load("map.npy") if path.exists("map.npy") else np.zeros(shape=[360,360])
state="auto_map_astar"  # "manual_map", "auto_map_astar", "auto_map_rrt"
# state = "vision"
current_path_world = []  # List of waypoints (in world coords)
current_waypoint_index = 0
# Define the planning update frequency (e.g., every 50 timesteps)
planning_interval = 1000  
planning_counter = 950
obstacle_thresh = 0.5
location_check_interval=100
location_check_counter=0
prev_location = None
# ------------------------------------------------------------------
# Helper Functions
x_dim= 29.0
y_dim= 15.0
scale_x = 360 / x_dim   # ~12.857 pixels per meter for x
scale_y = 360 / y_dim  # ~25.714 pixels per meter for y

UNEXPLORED = 0
FREE       = 1
OBSTACLE   = 2

image_data=None

gripper_status="closed"
print("=== Running Grocery Shopper...")
# Main Loop
while robot.step(timestep) != -1:
    
    pose_y = -gps.getValues()[1]
    pose_x = -gps.getValues()[0]

    # print(f'x: {-gps.getValues()[0]} y: {-gps.getValues()[1]}')

    n = compass.getValues()
    rad = (math.atan2(n[0], n[1]))
    pose_theta = rad
    print("Pose_x: ", pose_x, "Pose_y: ", pose_y, "Pose_theta: ", pose_theta)
    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]
    robot_gx = int((pose_x + x_dim/2) * scale_x)
    robot_gy = int((y_dim/2 + pose_y) * scale_y)

    # process each lidar beam
    for i, rho in enumerate(lidar_sensor_readings):
        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue

        alpha = lidar_offsets[i]
        # robot‐centric hit point
        rx = -math.cos(alpha)*rho
        ry =  math.sin(alpha)*rho
        # world coords
        wx =  math.cos(pose_theta)*rx - math.sin(pose_theta)*ry + pose_x
        wy =  math.sin(pose_theta)*rx + math.cos(pose_theta)*ry + pose_y

        gx = int((wx + x_dim/2) * scale_x)
        gy = int((y_dim/2 + wy) * scale_y)

        if not (0 <= gx < map.shape[0] and 0 <= gy < map.shape[1]):
            continue
        map[int((pose_x + x_dim/2) * scale_x), int((y_dim/2 + pose_y) * scale_y)] = FREE
        # trace the beam
        ray = bresenham(robot_gx, robot_gy, gx, gy)
        # mark all but the last cell as FREE (if they were still UNEXPLORED)
        for cx, cy in ray[:-1]:
            if map[cx, cy] == UNEXPLORED:
                map[cx, cy] = FREE
        # finally, mark the endpoint as OBSTACLE (overwrite anything)
        map[gx, gy] = OBSTACLE
    
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
            y_robot = int((y_dim/2+ pose_y) * scale_y)
            x_robot = int((pose_x + x_dim/2) * scale_x)
            np.save("map.npy", map)
            map_display=(map*256**2+map*256+map)*255
            plt.scatter(y_robot,x_robot,marker='x',color='red')
            cmap = ListedColormap(['lightgray','lightgreen','black'])
            plt.figure(figsize=(6,6))
            plt.imshow(map,cmap=cmap,vmin=0, vmax=2,origin='upper',interpolation='nearest')
            plt.colorbar( ticks=[0,1,2], label='0=unexplored, 1=free, 2=obstacle')
            plt.savefig("map.png")
            np.save("map.npy", map)
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
        print("pose_x: ", pose_x, "pose_y: ", pose_y, "pose_theta: ", pose_theta)
            
    if state=="auto_map_astar":
        planning_counter += 1
        location_check_counter += 1
        # binary obstacle map for A* (1=obstacle, 0=free/unknown)
        bmap = (map == OBSTACLE).astype(np.uint8)
        if not np.all(np.isin(bmap, [0, 1])):
            print("Error in bmap: contains values other than 0 and 1")
            break
        # robot’s grid cell (row, col)
        rrow = int((y_dim/2 + pose_y) * scale_y)
        rcol = int((pose_x + x_dim/2) * scale_x)
        robot_cell = (rcol, rrow)
        
        if location_check_counter >= location_check_interval:
            location_check_counter = 0
            # check if robot has moved significantly
            if prev_location is not None:
                dx = pose_x - prev_location[0]
                dy = pose_y - prev_location[1]
                dtheta = pose_theta - prev_location[2]
                if math.hypot(dx, dy) < 0.1 and abs(dtheta) < 0.1:
                    print("Robot stuck, backing up one step=========================================================")
                    # give it a quick jolt backward or rotate
                    robot_parts["wheel_left_joint"].setVelocity(-0.5 * MAX_SPEED)
                    robot_parts["wheel_right_joint"].setVelocity(-0.5 * MAX_SPEED)
                    # then skip planning this cycle so it actually moves
                    prev_location = (pose_x, pose_y, pose_theta)
                    continue  # back to robot.step()
            prev_location = (pose_x, pose_y, pose_theta)

        # replan when interval elapses or path exhausted
        if planning_counter >= planning_interval or current_waypoint_index >= len(current_path_world):
        # if current_waypoint_index >= len(current_path_world):
            planning_counter = 0
            # collect all unexplored cells
            configured_map = np.zeros(bmap.shape)
            obstacle_detected = np.argwhere(bmap==1)
            configured_map[obstacle_detected[:,0], obstacle_detected[:,1]] = 1
            footprint_radius = 8
            for x, y in obstacle_detected:
                x_min, x_max = max(0, x - footprint_radius), min(bmap.shape[0], x + footprint_radius)
                y_min, y_max = max(0, y - footprint_radius), min(bmap.shape[1], y + footprint_radius)
                configured_map[x_min:x_max, y_min:y_max] = 1
            if not np.all(np.isin(configured_map, [0, 1])):
                print("Error in configured map: contains values other than 0 and 1")
                break
            unexplored_cells = list(zip(*np.where((map == UNEXPLORED) & (configured_map == 0))))
            if not unexplored_cells:
                print("Exploration complete!")
                vL = vR = 0
                break
            # pick a random unexplored and attempt A*
            random.shuffle(unexplored_cells)
            target_cell = None
            for cell in unexplored_cells:
                path_grid = a_star(configured_map, robot_cell, cell)
                for (r, c) in path_grid:
                    if configured_map[r, c] == 1:
                        print(f"⚠️ Path goes through an obstacle at grid cell {(r,c)}")
                        break
                if len(path_grid) > 1:
                    cmap = ListedColormap(['purple','yellow'])
                    plt.imshow(configured_map,origin='upper',cmap=cmap,vmin=0, vmax=1)
                    if path_grid:
                        rows, cols = zip(*path_grid)
                        plt.plot(cols, rows,color='red',label='Planned Path')
                        np.save("configured_map.npy", configured_map)
                        plt.savefig("configured_map.png")
                    plt.close()
                    target_cell = cell
                    break
                else:
                    # mark unreachable as obstacle to avoid retry
                    map[cell] = OBSTACLE

            if target_cell is None:
                print("No reachable unexplored cells remaining!")
                vL = vR = 0
                break

            # convert to world waypoints, skipping the start
            current_path_world = [grid_to_world((c,r), scale_x, scale_y)for (c,r) in path_grid[1:]]
            current_waypoint_index = 0
            print(f"Planned path to random unexplored {target_cell}, length {len(path_grid)}")
            cmap = ListedColormap(['lightgray','lightgreen','black'])
            plt.figure(figsize=(6,6))
            plt.imshow(map,cmap=cmap,vmin=0, vmax=2,origin='upper',interpolation='nearest')
            if path_grid:
                rows, cols = zip(*path_grid)
                plt.plot(cols, rows,color='red',label='Planned Path')
            plt.colorbar( ticks=[0,1,2],label='0=unexplored, 1=free, 2=obstacle')
            plt.savefig("path.png")
            plt.close()
            np.save("map.npy", map)
        if current_path_world:
            if current_waypoint_index >= len(current_path_world):
                print("Reached end of path. Replanning...")
                current_path_world = []
                current_waypoint_index = 0
            else:
                wp = current_path_world[current_waypoint_index]
            wp = current_path_world[current_waypoint_index]
            x_goal,y_goal=wp
            print("Y_goal: ", y_goal, "X_goal: ", x_goal)
            rho=np.sqrt((x_goal - pose_x) ** 2 + (y_goal - pose_y) ** 2)  # Distance to the goal
            theta_g = np.arctan2(y_goal - pose_y, x_goal - pose_x)  # Angle to the goal
            alpha = theta_g - pose_theta-np.pi  # Difference from robot's heading
            if alpha<-np.pi:
                alpha += 2*np.pi
            vL=max(min(-8*alpha+12.56*rho, MAX_SPEED*0.7),-MAX_SPEED*0.7)
            vR=max(min(8*alpha+12.56*rho, MAX_SPEED*0.7),-MAX_SPEED*0.7)
            print(f"alpha={alpha:.2f}, rho={rho:.2f}")
            if rho<0.15: 
                if current_waypoint_index < len(current_path_world) - 1: 
                    current_waypoint_index += 1
                else: 
                    print("Goal reached")
                    planning_counter=planning_interval
                    vL=0
                    vR=0
        else:
            # no valid plan: rotate in place
            vL = -0.2 * MAX_SPEED
            vR =  0.2 * MAX_SPEED
            
    if state=="auto_map_rrt":
        planning_counter += 1
        location_check_counter += 1
        # binary obstacle map for A* (1=obstacle, 0=free/unknown)
        bmap = (map == OBSTACLE).astype(np.uint8)
        if not np.all(np.isin(bmap, [0, 1])):
            print("Error in bmap: contains values other than 0 and 1")
            break
        # robot’s grid cell (row, col)
        rrow = int((y_dim/2 + pose_y) * scale_y)
        rcol = int((pose_x + x_dim/2) * scale_x)
        robot_cell = (rcol, rrow)
        if location_check_counter >= location_check_interval:
            location_check_counter = 0
            # check if robot has moved significantly
            if prev_location is not None:
                dx = pose_x - prev_location[0]
                dy = pose_y - prev_location[1]
                dtheta = pose_theta - prev_location[2]
                if math.hypot(dx, dy) < 0.1 and abs(dtheta) < 0.1:
                    print("Robot stuck, backing up one step=========================================================")
                    # give it a quick jolt backward or rotate
                    robot_parts["wheel_left_joint"].setVelocity(-0.5 * MAX_SPEED)
                    robot_parts["wheel_right_joint"].setVelocity(-0.5 * MAX_SPEED)
                    # then skip planning this cycle so it actually moves
                    prev_location = (pose_x, pose_y, pose_theta)
                    continue  # back to robot.step()
            prev_location = (pose_x, pose_y, pose_theta)
        # replan when interval elapses or path exhausted
        if planning_counter >= planning_interval or current_waypoint_index >= len(current_path_world):
        # if current_waypoint_index >= len(current_path_world):
            planning_counter = 0
            # collect all unexplored cells
            configured_map = np.zeros(bmap.shape)
            obstacle_detected = np.argwhere(bmap==1)
            configured_map[obstacle_detected[:,0], obstacle_detected[:,1]] = 1
            footprint_radius = 8
            for x, y in obstacle_detected:
                x_min, x_max = max(0, x - footprint_radius), min(bmap.shape[0], x + footprint_radius)
                y_min, y_max = max(0, y - footprint_radius), min(bmap.shape[1], y + footprint_radius)
                configured_map[x_min:x_max, y_min:y_max] = 1
            if not np.all(np.isin(configured_map, [0, 1])):
                print("Error in configured map: contains values other than 0 and 1")
                break
            unexplored_cells = list(zip(*np.where((map == UNEXPLORED) & (configured_map == 0))))
            if not unexplored_cells:
                print("Exploration complete!")
                vL = vR = 0
                break
            # pick a random unexplored and attempt A*
            random.shuffle(unexplored_cells)
            target_cell = None
            for cell in unexplored_cells:
                def state_valid_w(pt):
                    # world coords --> grid indices, in the same order you used above for map[gx, gy]
                    gx = int((pt[0] + x_dim/2) * scale_x)
                    gy = int((y_dim/2 + pt[1]) * scale_y)
                    # boundary check
                    if gx < 0 or gx >= configured_map.shape[0] or gy < 0 or gy >= configured_map.shape[1]:
                        return False
                    # only free if configured_map[gx, gy] == 0
                    return configured_map[gx, gy] == 0
                # set up continuous start/goal
                start_w = np.array([pose_x, pose_y])
                goal_w  = grid_to_world(cell, scale_x, scale_y)
                # run RRT* to generate a tree
                bounds = np.array([[-x_dim/2, x_dim/2], [-y_dim/2, y_dim/2]])
                nodes = rrt_star(bounds, None, state_valid_w,start_w, goal_w,k=500, delta_q=0.5)
                # backtrack best node to form a path
                dists = [np.linalg.norm(n.point - goal_w) for n in nodes]
                best = nodes[np.argmin(dists)]
                path_pts = []
                curr = best
                while curr:
                    path_pts.append(curr.point)
                    curr = curr.parent
                path_pts.reverse()
                # only accept if we got more than the start point
                if len(path_pts) > 1:
                    target_cell = cell
                    best_path_pts = path_pts
                    cmap = ListedColormap(['purple','yellow'])
                    plt.imshow(configured_map,origin='upper',cmap=cmap,vmin=0, vmax=1)
                    if path_grid:
                        rows, cols = zip(*path_grid)
                        plt.plot(cols, rows, color='red', label='Planned Path')
                        np.save("configured_map.npy", configured_map)
                        plt.savefig("configured_map.png")
                    plt.close()
                    break
                else:
                    map[cell] = OBSTACLE
            if target_cell is None:
                print("No reachable unexplored cells remaining!")
                vL = vR = 0
                break
           # use continuous world coordinates directly
            # use the valid path you just found
            current_path_world   = best_path_pts[1:]
            current_waypoint_index = 0
            print(f"Planned path to random unexplored {target_cell}, length {len(best_path_pts)}")
            cmap = ListedColormap(['lightgray','lightgreen','black'])
            plt.figure(figsize=(6,6))
            plt.imshow(map,cmap=cmap,vmin=0, vmax=2,origin='upper',interpolation='nearest')
            if best_path_pts:
                # best_path_pts is a list of [wx,wy] in meters
                # convert each to grid‐pixel coordinates
                xs = [ (pt[0] + x_dim/2) * scale_x for pt in best_path_pts ]
                ys = [ (pt[1] + y_dim/2) * scale_y for pt in best_path_pts ]
                # plot in red
                plt.plot(ys,xs,linewidth=2,color='red',label='Planned Path')
            plt.colorbar(ticks=[0,1,2],label='0=unexplored, 1=free, 2=obstacle')
            plt.savefig("path.png")
            plt.close()
            np.save("map.npy", map)
        if current_path_world:
            if current_waypoint_index >= len(current_path_world):
                print("Reached end of path. Replanning...")
                current_path_world = []
                current_waypoint_index = 0
            else:
                wp = current_path_world[current_waypoint_index]
            wp = current_path_world[current_waypoint_index]
            x_goal,y_goal=wp
            print("Y_goal: ", y_goal, "X_goal: ", x_goal)
            rho=np.sqrt((x_goal - pose_x) ** 2 + (y_goal - pose_y) ** 2)  # Distance to the goal
            theta_g = np.arctan2(y_goal - pose_y, x_goal - pose_x)  # Angle to the goal
            alpha = theta_g - pose_theta-np.pi  # Difference from robot's heading
            if alpha<-np.pi:
                alpha += 2*np.pi
            vL=max(min(-8*alpha+12.56*rho, MAX_SPEED*0.7),-MAX_SPEED*0.7)
            vR=max(min(8*alpha+12.56*rho, MAX_SPEED*0.7),-MAX_SPEED*0.7)
            print(f"alpha={alpha:.2f}, rho={rho:.2f}")
            if rho<0.15: 
                if current_waypoint_index < len(current_path_world) - 1: current_waypoint_index += 1
                else: 
                    print("Goal reached")
                    planning_counter=planning_interval
                    vL=0
                    vR=0
        else:
            # no valid plan: rotate in place
            vL = -0.2 * MAX_SPEED
            vR =  0.2 * MAX_SPEED


    if state == "vision":
        vL = 0
        vR = 0
        image_data = camera.getImage()
        dark_yellow = np.array([20, 100, 100])
        light_yellow = np.array([30, 255, 255])
    # Webots images are 4-channel (BGRA), convert to 3-channel BGR for OpenCV
    if image_data:
        img = np.frombuffer(image_data, np.uint8).reshape(
            (camera.getHeight(), camera.getWidth(), 4)
        )
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv, dark_yellow, light_yellow)
        res = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 10:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(img_bgr, 'Yellow Object', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        cv2.imshow('Yellow Object Detection', img_bgr)
        yellow_objects = [c for c in contours if cv2.contourArea(c) > 10]
        if yellow_objects:
            print(f"Detected {len(yellow_objects)} yellow object(s)")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("vL:", vL, "vR: ", vR)
    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)