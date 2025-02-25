import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/abhilash-ts434/Documents/ros2_ws/src/install/my_py_pkg'
