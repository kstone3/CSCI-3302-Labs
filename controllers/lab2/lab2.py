"""
A Webots controller that implements an FSM for:
1) Following a wall on the left,
2) Turning right if an obstacle is in front,
3) Turning 180° upon first light detection and then following the wall on the right,
4) Stopping upon second light detection.

Place this file in your <webots_project>/controllers/ directory.
Then, in the Webots Scene Tree, set your robot's 'controller' field to 'wall_follower_controller'.
"""

from controller import Robot

# -----------------------------------------------------------------
# Constants
# -----------------------------------------------------------------
TIME_STEP = 64  # Adjust if needed; common for an e-puck in Webots

# Define states
FOLLOW_LEFT = 0
TURN_RIGHT = 1
TURN_180 = 2
FOLLOW_RIGHT = 3
STOP = 4

# Sensor thresholds (tune these for your robot/environment)
DIST_THRESHOLD_FRONT = 0.5      # Proximity sensor reading for "obstacle in front"
DIST_REFERENCE_LEFT  =0.5   # Desired left wall distance
DIST_REFERENCE_RIGHT =0.5    # Desired right wall distance
LIGHT_THRESHOLD      = 500.0     # Light detection threshold

# Timed turning durations (in milliseconds) – tune for your robot
TURN_RIGHT_DURATION  = 500       # how long to turn right ~90°
TURN_180_DURATION    = 1000      # how long to spin 180°

# Motor speeds
BASE_SPEED = 3.0    # forward speed for wall-following
TURN_SPEED = 2.0    # speed for turning in place

class WallFollowerController:
    def __init__(self):
        # Create Robot instance
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())

        # Current FSM state
        self.current_state = FOLLOW_LEFT

        # Motors
        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        self.left_motor.setPosition(float('inf'))   # velocity control mode
        self.right_motor.setPosition(float('inf'))  # velocity control mode
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # self.distance_sensors = {
        #     'left': self.robot.getDevice('ps5'),
        #     'front_left': self.robot.getDevice('ps7'),
        #     'front': self.robot.getDevice('ps0'),
        #     'front_right': self.robot.getDevice('ps1'),
        #     'right': self.robot.getDevice('ps2'),
        # }
        # for sensor in self.distance_sensors.values():
        #     sensor.enable(self.time_step)
        self.ps = []
        for i in range(8):
            sensor = self.robot.getDevice("ps" + str(i))
            sensor.enable(self.time_step)
            self.ps.append(sensor)
        # Initialize light sensors
        self.light_sensors = {
            'left': [self.robot.getDevice(f'ls{i}') for i in [0, 1, 2]],
            'right': [self.robot.getDevice(f'ls{i}') for i in [5, 6, 7]]
        }
        for sensors in self.light_sensors.values():
            for sensor in sensors:
                sensor.enable(self.time_step)

        # Track turn timing (if using timed approach)
        self.turn_start_time = 0
        self.turn_duration = 0

    def run(self):
        """ Main control loop. This is called repeatedly until the simulation ends. """
        while self.robot.step(self.time_step) != -1:
            # 1. Read sensors
            ps_values = [s.getValue() for s in self.ps]
            # If using separate light sensors, use them:
            # ls_values = [ls.getValue() for ls in self.ls]
            # Otherwise, you might try using ps in ambient mode or just do a placeholder:
            ls_values = ps_values  # Example if your sensors are configured to read light as well

            # 2. Check conditions
            front_obstacle = self.frontObstacleDetected(ps_values)
            light_detected = self.lightDetected(ls_values)

            # 3. State Machine logic
            if self.current_state == FOLLOW_LEFT:
                # Behavior
                self.followWallOnLeft(ps_values)

                # Transitions
                if front_obstacle:
                    self.current_state = TURN_RIGHT
                    self.turn_start_time = self.currentTimeMs()
                    self.turn_duration = TURN_RIGHT_DURATION
                elif light_detected:
                    self.current_state = TURN_180
                    self.turn_start_time = self.currentTimeMs()
                    self.turn_duration = TURN_180_DURATION

            elif self.current_state == TURN_RIGHT:
                # Check if still turning
                if self.turnInProgress():
                    self.performRightTurn()
                else:
                    # Done turning; go back to following left wall
                    self.current_state = FOLLOW_LEFT

            elif self.current_state == TURN_180:
                if self.turnInProgress():
                    self.perform180Turn()
                else:
                    # Done turning; now follow right wall
                    self.current_state = FOLLOW_RIGHT

            elif self.current_state == FOLLOW_RIGHT:
                self.followWallOnRight(ps_values)

                if light_detected:
                    # Stop on second detection
                    self.current_state = STOP

            elif self.current_state == STOP:
                self.stopRobot()
                # Remain here forever
                continue

            # 4. End of loop => the simulation step() will repeat

    # -----------------------------------------------------------------
    # Helper methods: check conditions
    # -----------------------------------------------------------------
    def frontObstacleDetected(self, ps_values):
        """
        Return True if there's an obstacle in front.
        For an e-puck, ps0 and ps7 are roughly front-left and front-right.
        Adjust if needed for your specific sensor arrangement.
        """
        if ps_values[0] > DIST_THRESHOLD_FRONT or ps_values[7] > DIST_THRESHOLD_FRONT:
            return True
        return False

    def lightDetected(self, ls_values):
        """
        Return True if light is detected by left, front, or right sensors.
        For an e-puck with 8 sensors, let's check indices 0,1,2 and 5,6,7.
        Adjust for your actual layout or dedicated light sensors.
        """
        relevant_indices = [0,1,2,5,6,7]
        sensor_sum = sum(ls_values[i] for i in relevant_indices)
        if sensor_sum > LIGHT_THRESHOLD:
            return True
        return False

    def currentTimeMs(self):
        """ Return the current simulation time in milliseconds. """
        return self.robot.getTime() * 1000.0

    def turnInProgress(self):
        """
        True if we haven't exceeded the assigned turn duration yet.
        """
        now_ms = self.currentTimeMs()
        return (now_ms - self.turn_start_time) < self.turn_duration

    # -----------------------------------------------------------------
    # Helper methods: actuation (behaviors in each state)
    # -----------------------------------------------------------------
    def followWallOnLeft(self, ps_values):
        """
        Simple left wall following.
        For e-puck, ps2 is often left side sensor. Check your config.
        """
        # e-puck's left sensors might be ps2, ps1, etc. Adjust as needed.
        left_dist = ps_values[2]
        error = DIST_REFERENCE_LEFT - left_dist

        # A simple P-control
        kp = 0.005
        turn_correction = kp * error

        # Adjust speeds
        left_speed = BASE_SPEED + turn_correction
        right_speed = BASE_SPEED - turn_correction

        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

    def followWallOnRight(self, ps_values):
        """
        Simple right wall following.
        For e-puck, ps5 or ps6 might be the right side sensor. Check your config.
        """
        right_dist = ps_values[5]
        error = DIST_REFERENCE_RIGHT - right_dist

        kp = 0.005
        turn_correction = kp * error

        left_speed = BASE_SPEED - turn_correction
        right_speed = BASE_SPEED + turn_correction

        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

    def performRightTurn(self):
        """
        Turn in place to the right by driving left wheel forward
        and right wheel backward.
        """
        self.left_motor.setVelocity(TURN_SPEED)
        self.right_motor.setVelocity(-TURN_SPEED)

    def perform180Turn(self):
        """
        Same turning approach as performRightTurn, 
        but we do it for a longer duration to achieve ~180°.
        """
        self.left_motor.setVelocity(TURN_SPEED)
        self.right_motor.setVelocity(-TURN_SPEED)

    def stopRobot(self):
        """ Stop both motors. """
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

# -----------------------------------------------------------------
# Main function
# -----------------------------------------------------------------
def main():
    controller = WallFollowerController()
    controller.run()

if __name__ == "__main__":
    main()
