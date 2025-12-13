# CIGRL Docker Image
# Multi-stage build for ROS2 + PyTorch

FROM ros:humble-ros-base AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Install PyTorch (CPU version for smaller image)
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Create workspace
WORKDIR /cigrl_ws

# Copy ROS2 packages
COPY ros2_ws/src /cigrl_ws/src

# Build ROS2 workspace
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

# Copy remaining files
COPY configs /cigrl_ws/configs
COPY models /cigrl_ws/models
COPY scripts /cigrl_ws/scripts
COPY experiments /cigrl_ws/experiments

# Setup entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

# ===== GPU Stage =====
FROM base AS gpu

# Install CUDA-enabled PyTorch
RUN pip3 uninstall -y torch torchvision && \
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# ===== Development Stage =====
FROM base AS dev

# Install development tools
RUN apt-get update && apt-get install -y \
    vim \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python dev tools
RUN pip3 install \
    pytest \
    black \
    flake8 \
    mypy \
    ipython \
    jupyter
