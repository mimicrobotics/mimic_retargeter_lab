import sys

# Remove ROS and Mimic WS paths from sys.path
sys.path = [p for p in sys.path if "/opt/ros" not in p and "mimic_ws" not in p]
