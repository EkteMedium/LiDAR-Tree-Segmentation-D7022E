# D7022E
This is a project which is part of the course AI in the Process Industry and Automation (D7022E) at at Luleå University of Technology (LTU). It aims to investigate ground, tree, and low vegetation classification and tree instance segmentation using the FOR-instance dataset.

# Project Idea
Segmentation of individual trees based on LiDAR scans has proved a very difficult task due to the high structural complexity and diversity of forest environments. These LiDAR scans usually contain a mixture of different terrain, low vegetation, and overlapping trees, both making it difficult to separate trees from low vegetation but also grouping individual trees. Variations in forest density, species, and scan quality may further complicate the task, and limit the efficiency of current traditional algorithms.
This project therefore addresses the problem of automatic segmentation of forest LiDAR scans by solving two subtasks. First task is the removal of ground and low vegetation using supervised machine learning. The second task is segmentation of individual trees for further processing.

# Dataset
The models will be developed using the FOR-instance dataset which is described in “FOR-instance: a UAV laser scanning benchmark dataset for semantic and instance segmentation of individual trees” (Stefano, o.a., 2023). The dataset contains five curated UAV-based forest laser scans from a diverse range of global locations and forest types. The scans have been manually annotated into individual trees and semantic classes (e.g. stem, woody branches, live branches, terrain, low vegetation).

# Repository
This repository is organized into two sections classification and segmentation focusing on ground/low vegetation/tree classification and tree instance segmentation respectivly. The segmentation section is further divide into our DBSCAN/HDBSCAN segmentation and custom "Growth algorithm" segmentation