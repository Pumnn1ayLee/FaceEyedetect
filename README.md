# FaceEyedetect

🐼一个带有笑脸置信度的基于opencv的人脸检测项目代码

### 功能:眨眼检测，眨眼次数统计，微笑置信度，微笑后自动保存图片。

Python环境:Python 3.7.3

# 安装教程

# 安装opencv

1、查看python版本

cmd输入命令行,python。

2、根据对应python版本，去[链接在这](https://www.lfd.uci.edu/~gohlke/pythonlibs/),下载对应版本opencv。


3、下载 pip install wheel

4、下载 pip install numpy

5、最后下载 opencv

```pip3.8(你自己的版本号).exe install C:\Users\HP\AppData\Local\Programs\Python\Python38\Scripts\opencv_python-4.5.5-cp38-cp38-win_amd64.whl```(你自己的路径）
按回车键，会显示安装成功。

下面是以python 3.8为例对应的版本

![image](https://user-images.githubusercontent.com/93638514/222913425-b59ba53c-7880-4278-8ec1-f6ce0a68d999.png)

之后cmd输入python并输入以下命令:

```import cv2```

没有报错则安装成功

# 安装dlib库

1.下载官方网址 [链接在这](http://dlib.net/)，注意对应python版本号，不同版本不同文件

2.打开win系统cmd,进入dilb包解压文件夹路径；

3.安装dlib

```pip install dlib-19.17.99-cp37-cp37m-win_amd64.whl``` 别照着复制，后面那个文件是你自己对应下载的.whl文件

同样cmd输入python,输入以下命令

```import dlib```
   
 没有报错则安装成功

# 之后在代码里有

```predictor = dlib.shape_predictor("E://Download/shape_predictor_68_face_landmarks.dat.dat")```改成项目里这个文件的你的路径(已经在项目里)

# 最后安装依赖

```pip install -r requirements.txt```

# 可以开始使用




