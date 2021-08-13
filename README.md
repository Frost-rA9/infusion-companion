# Infusion-Companion

#### 本项目使用的数据集
1. 文件夹介绍
   - bottle_normal：包含多个角度、光线、背景的瓶子、液位以及人像的干扰。主要目的为：**瓶子定位**
     - bottle_blackliquid：**使用墨水作为液滴**，可以用于瓶子定位以及液位检测，备用数据集，暂时没有针对性使用
   - face_expression：包含了从bottle_normal中截获的**表情**roi，还有百度图片中搜集而来的表情，制作而成的**表情二分类数据集**
   - liquid_classifier：所有的图像数据来自bottle_normal截获的roi，其中bottle_lower.txt文件中记录了哪些文件是upper或者Lower，用于**液位的二分类**
   - segmentation：尝试标注了整张bottle_normal的图像分割数据集，试图在**语义分割**中得到更好的鲁棒性，但实际效果较差。
   - WIRDE：数据集来自Multimedia Laboratory, Department of Information Engineering, The Chinese University of Hong Kong，用于人脸定位
2. 关于数据集开源
   - 本数据集开源，可以在学术目的上任意使用
   - 如需发表论文请加上：宁波大学科学技术学院
3. 下载地址：
   - 链接：https://pan.baidu.com/s/1iNWq_ZEoJGfTPqEQ_1iv-Q 
   - 提取码：NDKY 

