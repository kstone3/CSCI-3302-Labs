from turtle import left, right
from controller import Robot

class EPUCKController:
    
    def __init__(self):
    
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())
        
        # Initialize motors
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Initialize distance sensors
        self.distance_sensors = {
            'left': self.robot.getDevice('ps5'),
            'front_left': self.robot.getDevice('ps7'),
            'front_right': self.robot.getDevice('ps0'),
            'right': self.robot.getDevice('ps2'),
            'back_left' : self.robot.getDevice('ps4'),
            'back_right' : self.robot.getDevice('ps3'),
            
        }
        
        for sensor in self.distance_sensors.values(): sensor.enable(self.time_step)

        # Initialize light sensors
        self.light_sensors = {
            'left': [self.robot.getDevice(f'ls{i}') for i in [0, 1, 2]],
            'right': [self.robot.getDevice(f'ls{i}') for i in [5, 6, 7]]
        }
        
        for sensors in self.light_sensors.values():
            for sensor in sensors:
                sensor.enable(self.time_step)

        # State machine variables
        self.state = "FOLLOW_LEFT_WALL"
        self.wall_distance_threshold = 100
        self.wall_distance_max = 200  # Maxwimum acceptable distance from the wall
        self.obstacle_threshold = 300
        self.light_threshold = 1

    def read_distance_sensor(self, name): return (self.distance_sensors[name].getValue())


    def read_light_sensors(self):
        """
        Return True if light is detected from left, front, and right light sensors.
        """
        left_light_detected = all(sensor.getValue() < self.light_threshold for sensor in self.light_sensors['left'])
        right_light_detected = all(sensor.getValue() < self.light_threshold for sensor in self.light_sensors['right'])
        print(f"left_LightSensor: {left_light_detected:.2f}, right_LightSensor: {right_light_detected:.2f}")
        return left_light_detected and right_light_detected

    def set_motor_speeds(self, left_speed, right_speed):
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

    def follow_left_wall(self):
        print('Following wall on the left side')

        # Always use the left distance sensor to follow the wall
        wall_distance = self.read_distance_sensor('left')
        front_left_distance = self.read_distance_sensor('front_left')
        front_right_distance = self.read_distance_sensor('front_right')

        # Debugging sensor values
        # print(f"Wall Distance (left): {wall_distance:.2f}, Front Left: {front_left_distance:.2f}, Front Right: {front_right_distance:.2f}")

        # Check for obstacle in front or at an angle
        if (front_left_distance > self.obstacle_threshold or
            front_right_distance > self.obstacle_threshold):
            print("Obstacle detected, transitioning to AVOID_OBSTACLE.")
            self.state = "AVOID_OBSTACLE"
            return

        if left_wall_distance  < 64:
            print("Running off to nothing, transitioning to AVOID_NOTHING.")
            self.state = "AVOID_NOTHING"
            return

        #If the robot is too far from the wall, adjust to move closer
        if wall_distance < 100 and wall_distance > 75: #64 before
            print("Too far from the wall, adjusting path to move closer.")
            turn_duration = 0.035
            start_time = self.robot.getTime()
            while self.robot.getTime() - start_time < turn_duration:
                self.set_motor_speeds(-1.8, 1.8)  # Turn left slightly
                self.robot.step(self.time_step)
            return
        
        if wall_distance > 1200:
            print("Too close from the wall, adjusting path to move closer.")
            turn_duration = 0.035
            start_time = self.robot.getTime()
            while self.robot.getTime() - start_time < turn_duration:
                self.set_motor_speeds(1.8, -1.8)  # Turn right slightly
                self.robot.step(self.time_step)
            return
        
        else:
            left_speed = 3.0
            right_speed = 3.0

        # Debugging motor speeds
        # print(f"Motor Speeds: Left={left_speed:.2f}, Right={right_speed:.2f}")

        # Set motor speeds
        self.set_motor_speeds(left_speed, right_speed)
        
    def avoid_nothing(self):
        turn_duration = 1.1 
        start_time_buffer = 1.125
        end_time_buffer = 1.75
        print("Avoiding running off to nothing.")
        start_buffer_time = self.robot.getTime()
        while self.robot.getTime() - start_buffer_time < start_time_buffer:
            self.set_motor_speeds(2.0, 2.0)
            self.robot.step(self.time_step)
        start_time = self.robot.getTime()
        while self.robot.getTime() - start_time < turn_duration:
            self.set_motor_speeds(-2.0, 2.0)  # Turn left
            self.robot.step(self.time_step)
        end_buffer_time = self.robot.getTime()
        while self.robot.getTime() - end_buffer_time < end_time_buffer:
            self.set_motor_speeds(2.0, 2.0)
            self.robot.step(self.time_step)
        self.state = "FOLLOW_LEFT_WALL"

    def avoid_obstacle(self):
        turn_duration = 1.025
        print("Avoiding obstacle.")
        # Determine turn direction based on sensor readings
        left_wall_distance = self.read_distance_sensor('left')
        right_wall_distance = self.read_distance_sensor('right')

        # Default turn direction is right unless front_right detects a closer obstacle
        if right_wall_distance > left_wall_distance:
            print("Turning left to avoid obstacle.")
            start_time = self.robot.getTime()
            while self.robot.getTime() - start_time < turn_duration:
                self.set_motor_speeds(-2.0, 2.0)  # Turn left
                self.robot.step(self.time_step)
        else:
            print("Turning right to avoid obstacle.")
            while self.read_distance_sensor('front_right') > self.obstacle_threshold:
                start_time = self.robot.getTime()
                while self.robot.getTime() - start_time < turn_duration:
                    self.set_motor_speeds(2.0, -2.0)  # Turn right
                    self.robot.step(self.time_step)
        self.state = "FOLLOW_LEFT_WALL"
        
    def turn_around(self):
        # Perform a 180-degree turn
        turn_time = 2.0  # Adjust timing based on simulation
        start_time = self.robot.getTime()
        self.set_motor_speeds(-2.0, 2.0)
        while self.robot.getTime() - start_time < turn_time : self.robot.step(self.time_step)
        self.state = "FOLLOW_RIGHT_WALL"
        
    def follow_right_wall(self):
        print('Following wall on the right side')

        # Always use the left distance sensor to follow the wall
        wall_distance = self.read_distance_sensor('right')
        front_left_distance = self.read_distance_sensor('front_left')
        front_right_distance = self.read_distance_sensor('front_right')

        # Debugging sensor values
        # print(f"Wall Distance (right): {wall_distance:.2f}, Front Left: {front_left_distance:.2f}, Front Right: {front_right_distance:.2f}")

        # Check for obstacle in front or at an angle
        if (front_right_distance > self.obstacle_threshold or
            front_left_distance > self.obstacle_threshold):
            print("Obstacle detected, transitioning to AVOID_OBSTACLE.")
            self.state = "AVOID_RIGHT_OBSTACLE"
            return

        #If the robot is too far from the wall, adjust to move closer
        if wall_distance < 63:
            print("Too far from the wall, adjusting path to move closer.")
            turn_duration = 0.2
            start_time = self.robot.getTime()
            while self.robot.getTime() - start_time < turn_duration:
                self.set_motor_speeds(1.5, -1.5)  # Turn left slightly
                self.robot.step(self.time_step)
            return
        else:
            left_speed = 3.0
            right_speed = 3.0

        # Debugging motor speeds
        # print(f"Motor Speeds: Left={left_speed:.2f}, Right={right_speed:.2f}")

        # Set motor speeds
        self.set_motor_speeds(left_speed, right_speed)
        self.state = "FOLLOW_RIGHT_WALL"
    
    def avoid_right_obstacle(self):
        turn_duration = 1.3
        print("Avoiding obstacle.")
        # Determine turn direction based on sensor readings
        left_wall_distance = self.read_distance_sensor('left')
        right_wall_distance = self.read_distance_sensor('right')

        # Default turn direction is right unless front_right detects a closer obstacle
        if right_wall_distance > left_wall_distance:
            print("Turning left to avoid obstacle.")
            start_time = self.robot.getTime()
            while self.robot.getTime() - start_time < turn_duration:
                self.set_motor_speeds(-2.0, 2.0)  # Turn left
                self.robot.step(self.time_step)
        else:
            print("Turning right to avoid obstacle.")
            while self.read_distance_sensor('front_right') > self.obstacle_threshold:
                start_time = self.robot.getTime()
                while self.robot.getTime() - start_time < turn_duration:
                    self.set_motor_speeds(2.0, -2.0)  # Turn right
                    self.robot.step(self.time_step)

        self.state = "FOLLOW_RIGHT_WALL"

    def run(self):
        while self.robot.step(self.time_step) != -1:
            if self.state == "FOLLOW_LEFT_WALL":
                self.follow_left_wall()
                if self.read_light_sensors():
                    print("Light detected, transitioning to TURN_AROUND.")
                    self.state = "TURN_AROUND"
            elif self.state == "FOLLOW_RIGHT_WALL":
                self.follow_right_wall()
                if self.read_light_sensors():
                    print("Light detected, transitioning to TURN_AROUND.")
                    turn_time = 2.0  # Adjust timing based on simulation
                    start_time = self.robot.getTime()
                    self.set_motor_speeds(-2.0, 2.0)
                    while self.robot.getTime() - start_time < turn_time :
                        self.robot.step(self.time_step)
                    self.left_motor.setVelocity(0.0)
                    self.right_motor.setVelocity(0.0)
                    break
            elif self.state == "AVOID_NOTHING":
                self.avoid_nothing()
            elif self.state == "AVOID_OBSTACLE":
                self.avoid_obstacle()
            elif self.state == "AVOID_RIGHT_OBSTACLE":
                self.avoid_right_obstacle()
            elif self.state == "TURN_AROUND":
                print("Turning around.")
                self.turn_around()

if __name__ == "__main__":
    controller = EPUCKController()
    controller.run()