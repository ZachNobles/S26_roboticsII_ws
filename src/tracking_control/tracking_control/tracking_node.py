import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from tf2_ros import TransformException, Buffer, TransformListener
import numpy as np
import math

## Functions for quaternion and rotation matrix conversion
## The code is adapted from the general_robotics_toolbox package
## Code reference: https://github.com/rpiRobotics/rpi_general_robotics_toolbox_py
def hat(k):
    """
    Returns a 3 x 3 cross product matrix for a 3 x 1 vector

             [  0 -k3  k2]
     khat =  [ k3   0 -k1]
             [-k2  k1   0]

    :type    k: numpy.array
    :param   k: 3 x 1 vector
    :rtype:  numpy.array
    :return: the 3 x 3 cross product matrix
    """

    khat=np.zeros((3,3))
    khat[0,1]=-k[2]
    khat[0,2]=k[1]
    khat[1,0]=k[2]
    khat[1,2]=-k[0]
    khat[2,0]=-k[1]
    khat[2,1]=k[0]
    return khat

def q2R(q):
    """
    Converts a quaternion into a 3 x 3 rotation matrix according to the
    Euler-Rodrigues formula.
    
    :type    q: numpy.array
    :param   q: 4 x 1 vector representation of a quaternion q = [q0;qv]
    :rtype:  numpy.array
    :return: the 3x3 rotation matrix    
    """
    
    I = np.identity(3)
    qhat = hat(q[1:4])
    qhat2 = qhat.dot(qhat)
    return I + 2*q[0]*qhat + 2*qhat2
######################

def euler_from_quaternion(q):
    w=q[0]
    x=q[1]
    y=q[2]
    z=q[3]
    # euler from quaternion
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - z * x))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    return [roll,pitch,yaw]

class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking_node')
        self.get_logger().info('Tracking Node Started')
        
        # Current object pose
        self.obs_pose = None
        self.goal_pose = None
        
        # ROS parameters
        self.declare_parameter('world_frame_id', 'odom')

        # Create a transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Create publisher for the control command
        self.pub_control_cmd = self.create_publisher(Twist, '/track_cmd_vel', 10)
        # Create a subscriber to the detected object pose
        self.sub_detected_goal_pose = self.create_subscription(PoseStamped, 'detected_color_object_pose', self.detected_obs_pose_callback, 10)
        self.sub_detected_obs_pose = self.create_subscription(PoseStamped, 'detected_color_goal_pose', self.detected_goal_pose_callback, 10)

        # Create timer, running at 100Hz
        self.timer = self.create_timer(0.01, self.timer_update)

        self.start_pose = None
        self.state = 'go_to_goal'  # states: go_to_goal, return_home, done
        self.goal_reached_count = 0
        self.goal_stable_cycles = 5
    
    def detected_obs_pose_callback(self, msg):
        #self.get_logger().info('Received Detected Object Pose')
        
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        
        # TODO: Filtering
        # You can decide to filter the detected object pose here
        # For example, you can filter the pose based on the distance from the camera
        # or the height of the object
        # if np.linalg.norm(center_points) > 3 or center_points[2] > 0.7:
        #     return
        
        try:
            # Transform the center point from the camera frame to the world frame
            transform = self.tf_buffer.lookup_transform(odom_id,msg.header.frame_id,rclpy.time.Time(),rclpy.duration.Duration(seconds=0.1))
            t_R = q2R(np.array([transform.transform.rotation.w,transform.transform.rotation.x,transform.transform.rotation.y,transform.transform.rotation.z]))
            cp_world = t_R@center_points+np.array([transform.transform.translation.x,transform.transform.translation.y,transform.transform.translation.z])
        except TransformException as e:
            self.get_logger().error('Transform Error: {}'.format(e))
            return
        
        # Get the detected object pose in the world frame
        self.obs_pose = cp_world

    def detected_goal_pose_callback(self, msg):
        #self.get_logger().info('Received Detected Object Pose')
        
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        
        # TODO: Filtering
        # You can decide to filter the detected object pose here
        # For example, you can filter the pose based on the distance from the camera
        # or the height of the object
        # if np.linalg.norm(center_points) > 3 or center_points[2] > 0.7:
        #     return

        if np.linalg.norm(center_points) < 0.3:
            return
        
        try:
            # Transform the center point from the camera frame to the world frame
            transform = self.tf_buffer.lookup_transform(odom_id,msg.header.frame_id,rclpy.time.Time(),rclpy.duration.Duration(seconds=0.1))
            t_R = q2R(np.array([transform.transform.rotation.w,transform.transform.rotation.x,transform.transform.rotation.y,transform.transform.rotation.z]))
            cp_world = t_R@center_points+np.array([transform.transform.translation.x,transform.transform.translation.y,transform.transform.translation.z])
        except TransformException as e:
            self.get_logger().error('Transform Error: {}'.format(e))
            return
        
        # Get the detected object pose in the world frame
        self.goal_pose = cp_world
        
    def get_current_poses(self):
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        try:
            transform = self.tf_buffer.lookup_transform('base_footprint', odom_id, rclpy.time.Time())
            robot_world_x = transform.transform.translation.x
            robot_world_y = transform.transform.translation.y
            robot_world_z = transform.transform.translation.z
            robot_world_R = q2R([transform.transform.rotation.w,
                                transform.transform.rotation.x,
                                transform.transform.rotation.y,
                                transform.transform.rotation.z])
        except TransformException as e:
            self.get_logger().error('Transform error: ' + str(e))
            return None, None

        obstacle_pose = None
        goal_pose = None

        if self.obs_pose is not None:
            obstacle_pose = robot_world_R @ self.obs_pose + np.array([robot_world_x, robot_world_y, robot_world_z])
        if self.goal_pose is not None:
            goal_pose = robot_world_R @ self.goal_pose + np.array([robot_world_x, robot_world_y, robot_world_z])

        return obstacle_pose, goal_pose

    def get_robot_pose(self):
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        try:
            transform = self.tf_buffer.lookup_transform(odom_id, 'base_footprint', rclpy.time.Time())
        except TransformException as e:
            self.get_logger().error('Robot pose transform error: ' + str(e))
            return None, None

        robot_world_x = transform.transform.translation.x
        robot_world_y = transform.transform.translation.y
        q = np.array([transform.transform.rotation.w,
                      transform.transform.rotation.x,
                      transform.transform.rotation.y,
                      transform.transform.rotation.z])
        _, _, yaw = euler_from_quaternion(q)
        return np.array([robot_world_x, robot_world_y]), yaw

    def get_return_pose(self):
        if self.start_pose is None:
            return None

        robot_pose, yaw = self.get_robot_pose()
        if robot_pose is None or yaw is None:
            return None

        delta = self.start_pose - robot_pose
        c = math.cos(yaw)
        s = math.sin(yaw)
        return np.array([c * delta[0] + s * delta[1],
                         -s * delta[0] + c * delta[1],
                         0.0])

    def timer_update(self):
        if self.start_pose is None:
            robot_pose, _ = self.get_robot_pose()
            if robot_pose is not None:
                self.start_pose = robot_pose
                self.get_logger().info(f'Captured start pose: {self.start_pose.tolist()}')

        current_obs_pose, detected_goal_pose = self.get_current_poses()

        if self.state == 'go_to_goal':
            if detected_goal_pose is None:
                cmd_vel = Twist()
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.0
                self.pub_control_cmd.publish(cmd_vel)
                return
            current_goal_pose = detected_goal_pose
        elif self.state == 'return_home':
            current_goal_pose = self.get_return_pose()
            if current_goal_pose is None:
                cmd_vel = Twist()
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.0
                self.pub_control_cmd.publish(cmd_vel)
                return
        else:
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.pub_control_cmd.publish(cmd_vel)
            return

        cmd_vel = self.controller(current_obs_pose, current_goal_pose)
        self.pub_control_cmd.publish(cmd_vel)
        #################################################
    
    def controller(self, current_obs_pose, current_goal_pose):
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0

        kp_linear = 0.6
        kp_angular = 1.2
        max_linear = 0.25
        max_angular = 1.0
        goal_threshold = 0.18
        return_threshold = 0.20
        avoid_zone = 0.8
        avoid_gain = 1.0

        goal_x = current_goal_pose[0]
        goal_y = current_goal_pose[1]
        distance_to_goal = math.hypot(goal_x, goal_y)
        angle_to_goal = math.atan2(goal_y, goal_x)

        threshold = goal_threshold if self.state == 'go_to_goal' else return_threshold

        if distance_to_goal < threshold:
            self.goal_reached_count += 1
            if self.goal_reached_count >= self.goal_stable_cycles:
                if self.state == 'go_to_goal':
                    self.state = 'return_home'
                    self.get_logger().info('Goal reached, switching to return_home')
                elif self.state == 'return_home':
                    self.state = 'done'
                    self.get_logger().info('Returned to start, stopping')
                self.goal_reached_count = 0
            return cmd_vel

        self.goal_reached_count = 0
        cmd_vel.linear.x = kp_linear * distance_to_goal
        cmd_vel.angular.z = kp_angular * angle_to_goal

        if current_obs_pose is not None:
            obs_x = current_obs_pose[0]
            obs_y = current_obs_pose[1]
            obs_dist = math.hypot(obs_x, obs_y)

            if obs_dist < avoid_zone and obs_x > 0.0:
                turn_direction = -1.0 if obs_y > 0.0 else 1.0
                cmd_vel.angular.z += turn_direction * avoid_gain * (avoid_zone - obs_dist)
                cmd_vel.linear.x *= 0.5

        cmd_vel.linear.x = min(max(cmd_vel.linear.x, 0.0), max_linear)
        cmd_vel.angular.z = min(max(cmd_vel.angular.z, -max_angular), max_angular)

        return cmd_vel
    
        ############################################

def main(args=None):
    # Initialize the rclpy library
    rclpy.init(args=args)
    # Create the node
    tracking_node = TrackingNode()
    rclpy.spin(tracking_node)
    # Destroy the node explicitly
    tracking_node.destroy_node()
    # Shutdown the ROS client library for Python
    rclpy.shutdown()

if __name__ == '__main__':
    main()
