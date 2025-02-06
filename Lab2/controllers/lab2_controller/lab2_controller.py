"""csci3302_lab2 controller."""

# You may need to import some classes of the controller module.
import math
from controller import Robot, Motor, DistanceSensor
# import os

# Ground Sensor Measurements under this threshold are black
# measurements above this threshold can be considered white.
# TODO: Fill this in with a reasonable threshold that separates "line detected" from "no line detected"
GROUND_SENSOR_THRESHOLD = 0

# These are your pose values that you will update by solving the odometry equations
pose_x = 0
pose_y = 0
pose_theta = 0

# Index into ground_sensors and ground_sensor_readings for each of the 3 onboard sensors.
LEFT_IDX = 0
CENTER_IDX = 1
RIGHT_IDX = 2

# create the Robot instance.
robot = Robot()

# ePuck Constants
EPUCK_AXLE_DIAMETER = 0.053 # ePuck's wheels are 53mm apart.
EPUCK_MAX_WHEEL_SPEED = 0 # TODO: To be filled in with ePuck wheel speed in m/s
MAX_SPEED = 6.28

# get the time step of the current world.
SIM_TIMESTEP = int(robot.getBasicTimeStep())

# Initialize Motors
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

robot_state='speed_measurement'

# Initialize and Enable the Ground Sensors
gsr = [0, 0, 0]
ground_sensors = [robot.getDevice('gs0'), robot.getDevice('gs1'), robot.getDevice('gs2')] #front left, front center, front right
for gs in ground_sensors:
    gs.enable(SIM_TIMESTEP)

# Allow sensors to properly initialize
for i in range(10): robot.step(SIM_TIMESTEP)  

vL = 0 # TODO: Initialize variable for left speed
vR = 0 # TODO: Initialize variable for right speed

# Main Control Loop:
og_start_time=robot.getTime()
black_line_start_time = None  
while robot.step(SIM_TIMESTEP) != -1:

    # Read ground sensor values
    for i, gs in enumerate(ground_sensors): gsr[i] = gs.getValue()

    # print(gsr) # TODO: Uncomment to see the ground sensor values!
    if robot_state=="speed_measurement":
        start_time=robot.getTime()
        while -1<start_time-robot.getTime()<=0:
            leftMotor.setVelocity(MAX_SPEED)
            rightMotor.setVelocity(MAX_SPEED)
            robot.step(SIM_TIMESTEP)
        if all(gs.getValue() < 400 for gs in ground_sensors):
            leftMotor.setVelocity(0)
            rightMotor.setVelocity(0)
            robot_state='line_follower'
            time_elapsed=robot.getTime()-og_start_time
            distance_traveled = EPUCK_AXLE_DIAMETER/2 * MAX_SPEED * time_elapsed
            speed = distance_traveled / time_elapsed
            EPUCK_MAX_WHEEL_SPEED=speed
            print("Time Elapsed", time_elapsed)
            print("Speed: ",speed)
    if robot_state=='none': break
    
    # Hints: 
    #
    # 1) Setting vL=MAX_SPEED and vR=-MAX_SPEED lets the robot turn
    # right on the spot. vL=MAX_SPEED and vR=0.5*MAX_SPEED lets the
    # robot drive a right curve.
    #
    # 2) If your robot "overshoots", turn slower.
    #
    # 3) Only set the wheel speeds once so that you can use the speed
    # that you calculated in your odometry calculation.
    #
    # 4) Disable all console output to simulate the robot superfast
    # and test the robustness of your approach.
    #

    # TODO: Insert Line Following Code Here 
    if robot_state == "line_follower":
        if gsr[1] < 800:  # Center sensor detects the line
            vL = MAX_SPEED
            vR = MAX_SPEED
        elif gsr[0] < 800:  # Left sensor detects the line
            vL = -MAX_SPEED
            vR = MAX_SPEED
        elif gsr[2] < 800:  # Right sensor detects the line
            vL = MAX_SPEED
            vR = -MAX_SPEED
        else:  # None of the sensors detect the line
            vL = -MAX_SPEED
            vR = MAX_SPEED
    
    # TODO: Call update_odometry Here
    #THIS DOESN"T QUITE WORK
    vL_mps = (vL / MAX_SPEED) * EPUCK_MAX_WHEEL_SPEED
    vR_mps = (vR / MAX_SPEED) * EPUCK_MAX_WHEEL_SPEED
    d = (vL_mps + vR_mps) / 2.0 * SIM_TIMESTEP/1000
    d_theta = ((vR_mps - vL_mps) / EPUCK_AXLE_DIAMETER / 4) * SIM_TIMESTEP/1000
    pose_x += d * math.cos(pose_theta)
    pose_y -= d * math.sin(pose_theta)
    pose_theta += d_theta
    #pose_theta = (pose_theta + math.pi) % (2 * math.pi) - math.pi

    # Hints:
    #
    # 1) Divide vL/vR by MAX_SPEED to normalize, then multiply with
    # the robot's maximum speed in meters per second. 
    #
    # 2) SIM_TIMESTEP tells you the elapsed time per step. You need
    # to divide by 1000.0 to convert it to seconds
    #
    # 3) Do simple sanity checks. In the beginning, only one value
    # changes. Once you do a right turn, this value should be constant.
    #
    # 4) Focus on getting things generally right first, then worry
    # about calculating odometry in the world coordinate system of the
    # Webots simulator first (x points down, y points right)

    
    # TODO: Insert Loop Closure Code Here
    all_black = all(gs.getValue() < 400 for gs in ground_sensors)
    if all_black:
        if black_line_start_time is None:
            black_line_start_time = robot.getTime()  # Start timer
        elif robot.getTime() - black_line_start_time > 0.1:  # 0.1s threshold
            pose_x = 0
            pose_y = 0
            pose_theta = 0
            print("Pose reset after detecting black for 0.1 seconds!")
    else:
        black_line_start_time = None
    # Hints:
    #
    # 1) Set a flag whenever you encounter the line
    #
    # 2) Use the pose when you encounter the line last 
    # for best results
    
    
    print("Current pose: [%5f, %5f, %5f]" % (pose_x, pose_y, pose_theta))
    leftMotor.setVelocity(vL)
    rightMotor.setVelocity(vR)
