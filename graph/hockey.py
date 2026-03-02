import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 20
self_link = [(i, i) for i in range(num_node)]

# SKELETON DEFINITION
# 0-16: Body
# 17-19: Stick (Connected to both wrists)

skeleton_edges = [
    # Head
    (2, 1),   # nose -> left_ear
    (2, 0),   # nose -> right_ear
    (0, 1),   # right_ear -> left_ear
    
    # Torso / Shoulders
    (3, 4),   # right_shoulder -> left_shoulder
    (3, 7),   # right_shoulder -> right_elbow (Index 7)
    (7, 10),  # right_elbow -> right_wrist (Index 10)
    
    (4, 9),   # left_shoulder -> left_elbow (Index 9)
    (9, 8),   # left_elbow -> left_wrist (Index 8)
    
    (3, 5),   # right_shoulder -> right_hip
    (4, 6),   # left_shoulder -> left_hip
    (5, 6),   # right_hip -> left_hip
    
    # Legs
    (5, 11),  # right_hip -> right_knee
    (11, 14), # right_knee -> right_ankle
    (14, 15), # right_ankle -> right_foot_tip
    
    (6, 12),  # left_hip -> left_knee
    (12, 13), # left_knee -> left_ankle
    (13, 16), # left_ankle -> left_foot_tip
    
    # Stick → Body connection (both hands grip the stick)
    (8, 17),  # left_wrist -> stick_top
    (10, 17), # right_wrist -> stick_top

    # Stick
    (17, 18), # stick_top -> stick_bottom
    (18, 19)  # stick_bottom -> stick_tip
]

# Use strictly the edges defined above
inward_ori_index = skeleton_edges
inward = inward_ori_index
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        
        # Generate the Binary Outward Matrix (Required by LST Model)
        self.A_outward_binary = tools.get_adjacency_matrix(outward, num_node)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        elif labeling_mode == 'spatial_hockey':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError(f"Labeling mode {labeling_mode} not supported.")
        return A
