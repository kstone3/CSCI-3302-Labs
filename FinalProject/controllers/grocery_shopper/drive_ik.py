# import translation
import math
import numpy as np
from typing import List, Tuple

class Drive_IK_Controller():
    def __init__(s, gps, compass, AXLE_LEN, RAD, MAX_SPEED):
        s.gps = gps
        s.compass = compass
        s.AXLE_LENGTH = AXLE_LEN
        s.RADIUS = RAD
        s.MAX_SPEED = MAX_SPEED

    def set_state(s, state: int):
        if state < 0:
            s.state = 0
        elif state <= s.last_point:
            s.state = state
        elif state > s.last_point:
            s.state = s.last_point

    def get_state(s):
        return s.state

    def start_path(s, waypoints, pose_x,pose_y,pose_theta):
        s.state = 10
        
        s.last_point = len(waypoints) - 1
        print("# of waypoints: ", s.last_point+1)
        
        #Replace the last point in the path to be an extention of the path, this is necessary to keep the robot moving till it reaches the goal
        # print(waypoints, waypoints.shape, path, path.shape)
        xdiff = (waypoints[s.last_point][0] - waypoints[s.last_point - 15][0])
        ydiff = (waypoints[s.last_point][1] - waypoints[s.last_point - 15][1])
        waypoints[s.last_point][0] += xdiff
        waypoints[s.last_point][1] += ydiff

        s.scaled_waypoints = []
        for point in waypoints: 
            x = pose_x + math.cos(pose_theta) * point[0] - math.sin(pose_theta) * point[1]
            y = pose_y + math.sin(pose_theta) * point[0] + math.cos(pose_theta) * point[1]
            # print(point[2], np.mod(point[2],(2*math.pi)))
            s.scaled_waypoints.append([x, y, np.mod(point[2],(2*math.pi))])

    # will return None when end of path has been reached
    def path_step(s, pose_x,pose_y,pose_theta):
        if s.state < s.last_point:
            # print(s.state, s.scaled_waypoints[s.state])
            # Ground truth pose
            # pose_x, pose_y, pose_theta = translation.get_current_position(s.gps,s.compass)

            # print("loc: ", pose_x, pose_y, pose_theta)

            #calcute the errors 
            errorx = pose_x - s.scaled_waypoints[s.state][0]
            errory = pose_y - s.scaled_waypoints[s.state][1]

            # print("pre-err: ", errorx, errory)

            dist_error = np.linalg.norm([errorx, errory])
            
            # calc both sides of the -pi,pi discontinuity and take the min
            # bear_error = translation.get_bearing_difference(s.scaled_waypoints[s.state], (pose_x,pose_y,pose_theta))
            target = s.scaled_waypoints[s.state]
            target_x, target_y = target[0], target[1]
            desired_bearing = math.atan2(target_y - pose_y, target_x - pose_x)

            # Compute the bearing error (difference between desired bearing and current orientation)
            bear_error = desired_bearing - pose_theta

            # Normalize the bearing error to be within [-pi, pi]
            while bear_error > math.pi:
                bear_error -= 2 * math.pi
            while bear_error < -math.pi:
                bear_error += 2 * math.pi
            
            # print("test_err ", math.atan2(errory,errorx))
            # print(get_rotated_bearing((-1,0,0),(0,0,math.pi/2)))
            
            # head_error = s.scaled_waypoints[s.state][2] - (pose_theta%(2*math.pi))
            head_error = min([s.scaled_waypoints[s.state][2]-(pose_theta%(2*math.pi)), (2*math.pi)+s.scaled_waypoints[s.state][2]-(pose_theta%(2*math.pi))], key=abs)

            # print("errors: ",dist_error,bear_error,head_error)

            p1 = 0.7 # dist
            p2 = 0.5 # bearing
            p3 = 0.3 # heading

            # IK calculations
            x_dot_ik = p1*dist_error
            theta_dot_ik = p2*bear_error + p3*head_error
            
            #calc wheel speeds at full values
            phi_l = (x_dot_ik - (s.AXLE_LENGTH * theta_dot_ik)/2)/s.RADIUS
            phi_r = (x_dot_ik + (s.AXLE_LENGTH * theta_dot_ik)/2)/s.RADIUS
            
            # print("phi: ", phi_l, phi_r) 

            #if we have reached a goal move onto the next    
            if dist_error < .3:
                #Stop if we reach our desitnation
                if s.state >= s.last_point:
                    print("Path complete")
                    return None,None
                # phi_l = 0
                # phi_r = 0
                print(f"Waypoint {s.state} reached!")
                # print(int(scaled_waypoints[s.state][0]),int(scaled_waypoints[s.state][1]))
                s.state += 15  # only look at every 15th point
                if s.state >= s.last_point:
                    s.state = s.last_point
                    
            # Scale motor values to be within MAX_SPEED
            elif abs(phi_l) >= abs(phi_r) and abs(phi_l) > 0.1:
                ratio = abs(phi_r/phi_l) 
                
                if phi_l >= 0:
                    phi_l = s.MAX_SPEED
                else:
                    phi_l = -s.MAX_SPEED  
                
                if phi_r >= 0:
                    phi_r = s.MAX_SPEED * ratio
                else:
                    phi_r = -s.MAX_SPEED * ratio     
                    
            elif abs(phi_r) >= abs(phi_l) and abs(phi_r) > 0.1:
                ratio = abs(phi_l/phi_r)
                
                if phi_r >= 0:
                    phi_r = s.MAX_SPEED
                else:
                    phi_r = -s.MAX_SPEED  
                if phi_l >= 0:
                    phi_l = s.MAX_SPEED * ratio
                else:
                    phi_l = -s.MAX_SPEED * ratio   
                                
            
            vL = phi_l # Left wheel velocity in rad/s
            vR = phi_r # Right wheel velocity in rad/s

            return (vL,vR)
        else:
            return None,None
