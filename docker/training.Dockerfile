FROM python:3.10

WORKDIR /workspace

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy repo (build from clone with submodules: git submodule update --init --recursive)
COPY . .
RUN pip install -e simulators/pybullet-drones

CMD ["bash"]