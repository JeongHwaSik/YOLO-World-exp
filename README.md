# 🧪 YOLO-World-experiments 

## Notice
Thanks to Tianheng Cheng et al. for their excellent research on Real-Time Open-Vocabulary Object Detector: [YOLO-World](https://arxiv.org/pdf/2401.17270). This experimental code is entirely based on the official [YOLO-World repository](https://github.com/AILab-CVC/YOLO-World?tab=readme-ov-file) provided by the authors. See [here](https://github.com/AILab-CVC/YOLO-World?tab=readme-ov-file#getting-started) for installation and data preparation to get started!


## 👨🏻‍🔬 Experiments
### 1. Find out a bottleneck of the YOLO-World 
 To identify the bottleneck in YOLO-World, the average inference time for each block was measured using the LVIS-minival dataset, as shown in the figure below. Typically, when measuring benchmark FPS for the YOLO series, the NMS time is excluded. Additionally, in YOLO-World, text features are obtained offline during inference. Excluding the head time and text backbone time, the bottleneck was found in the neck block, which accounted for 60% of the total time. The neck block is where the image pyramid features(P3, P4, P5) from the image backbone block are fused with the text features. I hypothesized that this fusion might involve computational redundancy.

![Screenshot 2024-11-07 at 8 00 21 PM](https://github.com/user-attachments/assets/f11ee054-38cc-4c3b-a542-fb8377bf1d7d)

### 2. Computation redundancy in the neck block
