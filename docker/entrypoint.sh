#!/bin/bash
# CIGRL Docker Entrypoint

# Source ROS2
source /opt/ros/humble/setup.bash

# Source workspace
if [ -f /cigrl_ws/install/setup.bash ]; then
    source /cigrl_ws/install/setup.bash
fi

# Set environment variables
export CIGRL_ROOT=/cigrl_ws
export PYTHONPATH=$PYTHONPATH:$CIGRL_ROOT

# Execute command
exec "$@"
