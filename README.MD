# BTC
## BTC: A Binary and Triangle Combined Descriptor for 3D Place Recognition
## Introduction
**BTC** is a novel global and local combined descriptor for 3D place recognition. To achieve viewpoint invariance, we devise a global triangle descriptor composed of three side lengths of a triangle formed by three keypionts extracted from the accumulated point cloud. The triangle descriptor inherently exhibits viewpoint invariance, as the side lengths remain unchanged regardless of viewpoint or direction. While the triangle descriptor captures the global appearance of the point cloud, to further enhance its descriptive and discriminative capabilities, we develop a local descriptor called binary descriptor that encodes the local geometric information of each keypoint forming the triangle. This combination of global and local descriptors allows for accurate and robust place recognition, even in large-scale unstructured environments.

<div align="center">
    <div align="center">
        <img src="figs/system_pipeline_btc.png" width = 95% >
    </div>
    <font color=#a0a0a0 size=2>System overview of the Binary Triangle Combined (BTC) descriptor for place recognition in a SLAM system.</font>
</div>
  


### Developers:
[Chongjian Yuan 袁崇健](https://github.com/ChongjianYUAN)， [Jiarong Lin 林家荣](https://github.com/ziv-lin)


### Related paper
Related paper will be available on **arxiv** soon.  

### Related video
Our accompanying video is now available on **YouTube**.
<div align="center">
    <a href="https://youtu.be/zB4Xqi3-J9U" target="_blank"><img src="figs/video_cover.jpg" width=60% /></a>
</div>

## Codes & Datasets
Our paper is currently under review, and our code will be released once the paper is accepted.