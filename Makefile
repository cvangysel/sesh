SRC_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

all: $(SRC_DIR)/sesh/sesh_pb2.py

$(SRC_DIR)/sesh/sesh_pb2.py: $(SRC_DIR)/sesh/sesh.proto
	protoc -I=$(SRC_DIR) --python_out=$(SRC_DIR) $(SRC_DIR)/sesh/sesh.proto