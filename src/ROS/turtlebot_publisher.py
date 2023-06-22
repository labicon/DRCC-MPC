import julia
j = julia.Julia()
from julia import Main
Main.include("../ros_distributionally_robust_controller.jl")
import rospy
from geometry_msgs.msg import Twist, PoseArray

class TrutleBotDRCController():
    def __init__(self):
        # initialize the node
        rospy.init_node('turtlebot_controller', anonymous=True)

        # tell user how to stop TurtleBot
        rospy.loginfo("To stop TurtleBot CTRL + C")
        rospy.on_shutdown(self.shutdown)

        # subscribe pedestrian and turtlebot positions
        self.state_sub = rospy.Subscriber('/turtlebot_state', PoseArray, self.state_process, queue_size=1)
        
        # Publisher to control the robot's speed
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.control_rate = rospy.Rate(10)  # 10hz

    def state_process(self, msg):
        

    def publish(self, forward_vel, angular_vel):
        # publish the velocity
        move_cmd = Twist()
        move_cmd.linear.x = forward_vel
        move_cmd.angular.z = angular_vel

    def shutdown(self):
        # stop turtlebot
        rospy.loginfo("Stop TurtleBot")
        # a default Twist has linear.x of 0 and angular.z of 0.  So it'll stop TurtleBot
        self.cmd_vel.publish(Twist())
        # sleep just makes sure TurtleBot receives the stop command prior to shutting down the script
        rospy.sleep(1)


if __name__ == '__main__':
    node = DRCController()

    u = Main.get_control()
    try:
        node.publish(u[0], u[1])
    except:
        rospy.loginfo("node terminated.")