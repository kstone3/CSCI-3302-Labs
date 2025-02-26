"""csci3302_lab3 controller."""

# You may need to import some classes of the controller module.
import math

from sklearn.metrics import euclidean_distances
from controller import Robot, Motor, DistanceSensor, Supervisor
import numpy as np

pose_x=0
pose_y=0
pose_theta=0

# Index into ground_sensors and ground_sensor_readings for each of the 3 onboard sensors.
LEFT_IDX=0
CENTER_IDX=1
RIGHT_IDX=2

# create the Robot instance.
robot=Supervisor()

# ePuck Constants
EPUCK_AXLE_DIAMETER=0.053 # ePuck's wheels are 53mm apart.
EPUCK_MAX_WHEEL_SPEED=0.1257 # ePuck wheel speed in m/s
MAX_SPEED=6.28

robot_state="proportional_controller" # Initial state of the robot
robot_semi_state = 0 #turn_drive_turn_control solution
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

# Allow sensors to properly initialize
for i in range(10): robot.step(SIM_TIMESTEP)  

vL=0
vR=0

# Initialize gps and compass for odometry
gps=robot.getDevice("gps")
gps.enable(SIM_TIMESTEP)
compass=robot.getDevice("compass")
compass.enable(SIM_TIMESTEP)

# TODO: Find waypoints to navigate around the arena while avoiding obstacles
waypoints=[
    (-0.313756, -0.413751),
    (0.320071, -0.416924),
    (0.324869, -0.256266),
    (0.0497542, -0.0275227),
    (0.344084, 0.249577),
    (0.149131, 0.423142),
    (-0.303882, 0.416781),
    (-0.211567, 0.146211),
    (-0.305326, -0.0442482),
    (-0.310298, -0.253309)
]
# Index indicating which waypoint the robot is reaching next
index=0

# Get ping pong ball marker that marks the next waypoint the robot is reaching
marker=robot.getFromDef("marker").getField("translation")

# Main Control Loop:
while robot.step(SIM_TIMESTEP) != -1:
    # Set the position of the marker
    marker.setSFVec3f([waypoints[index][0], waypoints[index][1], 0.01])
    
    # Read ground sensor values
    for i, gs in enumerate(ground_sensors):
        gsr[i]=gs.getValue()

    # Read pose_x, pose_y, pose_theta from gps and compass
    pose_x=gps.getValues()[0]
    pose_y=gps.getValues()[1]
    pose_theta=np.arctan2(compass.getValues()[0], compass.getValues()[1])
    
    x_goal, y_goal=waypoints[index]
    rho=np.sqrt((x_goal - pose_x) ** 2 + (y_goal - pose_y) ** 2)  # Distance to the goal
    # alpha=np.arctan((y_goal - pose_y) / (x_goal - pose_x)) - pose_theta
    # alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
    # eta=np.arctan2(y_goal - pose_y, x_goal - pose_x) - pose_theta
    # eta = (eta + np.pi) % (2 * np.pi) - np.pi
    theta_g = np.arctan2(y_goal - pose_y, x_goal - pose_x)  # Angle to the goal
    alpha = theta_g - pose_theta  # Difference from robot's heading
    if alpha<-np.pi:
        alpha += 2*np.pi
    if index < len(waypoints) - 1:
        x_next, y_next = waypoints[index + 1]
    else:
        x_next, y_next = waypoints[0]
    theta_goal_orientation = np.arctan2(y_next - y_goal, x_next - x_goal)
    eta = theta_goal_orientation - pose_theta
    eta = (eta + np.pi) % (2 * np.pi) - np.pi
    print("Rho: ", rho)
    print("Alpha: ", alpha)
    print("Eta: ", eta)
    
    # TODO: controller
    if robot_state == "line_follower":
        if gsr[1]<700:  # Center sensor detects the line
            vL=MAX_SPEED / 4 
            vR=MAX_SPEED / 4 
        elif gsr[0]<700:  # Left sensor detects the line
            vL=-MAX_SPEED / 4
            vR=MAX_SPEED / 4
        elif gsr[2]<700:  # Right sensor detects the line
            vL=MAX_SPEED / 4
            vR=-MAX_SPEED / 4
        else:  # None of the sensors detect the line
            vL=-MAX_SPEED / 4
            vR=MAX_SPEED / 4

    if robot_state == "turn_drive_turn_control":
        # Rotate in place until the bearing error (α) is within tolerance.
        if (abs(alpha) > 0.1) and (robot_semi_state == 0):
            if alpha > 0:
                vL = -MAX_SPEED / 4
                vR = MAX_SPEED / 4
            else:
                vL = MAX_SPEED / 4
                vR = -MAX_SPEED / 4
        else:
            if robot_semi_state == 0:
                # Bearing error is small; proceed to drive forward.
                vL = 0
                vR = 0
                robot_semi_state = 1
            elif robot_semi_state == 1:
                # Drive forward while reducing position error (ρ).
                if (rho > 0.05):
                    vL =  MAX_SPEED / 2
                    vR =  MAX_SPEED / 2
                else:
                    robot_semi_state = 2
            elif robot_semi_state == 2:
                # Rotate to adjust the heading error (η).
                if abs(eta) > 0.1:
                    if eta > 0:
                        vL = -MAX_SPEED / 4
                        vR = MAX_SPEED / 4
                    else:
                        vL = MAX_SPEED / 4
                        vR = -MAX_SPEED / 4
                else:
                    # Finished the turn; stop and update the waypoint.
                    vL = 0
                    vR = 0
                    index = (index + 1) % len(waypoints)
                    robot_semi_state = 0
            
    if robot_state=="proportional_controller":
        vL=max(min(-8*alpha+12.56*rho, MAX_SPEED),-MAX_SPEED)
        vR=max(min(8*alpha+12.56*rho, MAX_SPEED),-MAX_SPEED)
        if rho<0.05: index = (index + 1) % len(waypoints)
        
    print("Current pose: [%5f, %5f, %5f]" % (pose_x, pose_y, pose_theta))
    leftMotor.setVelocity(vL)
    rightMotor.setVelocity(vR)