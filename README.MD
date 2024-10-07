# BTC
## BTC: A Binary and Triangle Combined Descriptor for 3D Place Recognition

## Introduction
**BTC** is a novel global and local combined descriptor for 3D place recognition. To achieve viewpoint invariance, we devise a global triangle descriptor composed of three side lengths of a triangle formed by three keypoints extracted from the accumulated point cloud. The triangle descriptor inherently exhibits viewpoint invariance, as the side lengths remain unchanged regardless of viewpoint or direction. While the triangle descriptor captures the global appearance of the point cloud, to further enhance its descriptive and discriminative capabilities, we develop a local descriptor called the binary descriptor that encodes the local geometric information of each keypoint forming the triangle. This combination of global and local descriptors allows for accurate and robust place recognition, even in large-scale unstructured environments.

<div align="center">
    <img src="figs/system_pipeline_btc.png" width="95%">
    <font color="#a0a0a0" size="2">System overview of the Binary Triangle Combined (BTC) descriptor for place recognition in a SLAM system.</font>
</div>

### Developers:
[Chongjian Yuan 袁崇健](https://github.com/ChongjianYUAN)， [Jiarong Lin 林家荣](https://github.com/ziv-lin)

### Related Paper
The related paper will be available on **arxiv** soon.

### Related Video
Our accompanying video is now available on **YouTube**.
<div align="center">
    <a href="https://youtu.be/zB4Xqi3-J9U" target="_blank"><img src="figs/video_cover.jpg" width="60%"/></a>
</div>

## Parameters Explanation

### Binary Descriptor Parameters
- `useful_corner_num`: Maximum number of useful corners.
- `plane_detection_thre`: Threshold for plane detection.
- `plane_merge_normal_thre`: Normal threshold for plane merging.
- `plane_merge_dis_thre`: Distance threshold for plane merging.
- `voxel_size`: Size of each voxel.
- `voxel_init_num`: Minimum number of points in a voxel for extraction.
- `proj_plane_num`: Number of projection planes.
- `proj_image_resolution`: Resolution of the projected image in meters.
- `proj_image_high_inc`: Height increment for the projected image in meters.
- `proj_dis_min`: Minimum distance to the projection plane.
- `proj_dis_max`: Maximum distance to the projection plane.
- `summary_min_thre`: Minimum number of points in one image grid.
- `line_filter_enable`: Enable line point filtering; turn off in environments lacking features (indoor and outdoor), otherwise enable to reduce computation

### Stable Triangle Descriptor Parameters
- `descriptor_near_num`: Number of neighboring points to save for each point when generating triangles.
- `descriptor_min_len`: Minimum edge length.
- `descriptor_max_len`: Maximum edge length.
- `max_constrait_dis`: Neighborhood range for non-maximum suppression.
- `triangle_resolution`: Edge length amplification coefficient.

### Candidate Search Parameters
- `skip_near_num`: Number of frames to skip.
- `candidate_num`: Maximum number of reference frames.
- `similarity_threshold`: BTC similarity coefficient.
- `rough_dis_threshold`: Threshold coefficient for BTC edge length distance.
- `normal_threshold`: Normal Threshold for plane-to-plane validation.
- `dis_threshold`: Distance threshold for plane-to-plane validation.
- `icp_threshold`: Threshold for plane-to-plane ICP to triggle loop.

## Example

### Place Recognition
To perform place recognition using the BTC descriptor, you need to prepare the data and then launch the ROS node with the following command:

1. **Data Preparation**:
   - Prepare the submap LiDAR point cloud directories containing the point cloud files in either `.bin` or `.pcd` format.
   - File naming format should follow this convention:
     - For PCD files: `000000.pcd`, `000001.pcd`, ..., up to your maximum submap count.
     - For BIN files: similarly named as `000000.bin`, `000001.bin`, etc.
   - Prepare a `pose.txt` file with the following format:
     ```
     Submap_id pos_x pos_y pos_z quat_x quat_y quat_z quat_w
     ```
     where:
     - `Submap_id` is the identifier for the submap.
     - `pos_x`, `pos_y`, `pos_z` are the coordinates of the submap's position.
     - `quat_x`, `quat_y`, `quat_z`, `quat_w` are the quaternion components representing the orientation.

2. **Launch the Node**:
   Run the following command to start the place recognition process:

   ```bash
   roslaunch btc_desc place_recognition.launch
   ```

### More Examples
More examples to be released soon:
- **Multi-Session Place Recognition**
- **Global Relocalization**
- **Pose Graph Optimization (PGO)**
- **Multi-Map Merge**
