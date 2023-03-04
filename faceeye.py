import dlib
import cv2
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time


def eye_aspect_ratio(eye):
    # 计算眼部垂直方向上2组关键点的欧式距离
    A = dist.enclidean(eye[1] , eye[5])
    B = dist.enclidean(eye[2] , eye[4])

    # 计算眼部水平方向上1组关键点欧式距离
    C = dist.enclidean(eye[0] , eye[3])

    # 计算EAR
    ear = (A + B) / (2.0 * C)

    return ear

# 眨眼次数
TOTAL = 0

# 数据帧计数器
COUNTER = 0

# 初始化dlib人脸检测器
detector = dlib.get_frontal_face_detector()


# 脸部关键点检测器
predictor = dlib.shape_predictor("E://Download/shape_predictor_68_face_landmarks.dat.dat")

# 得到左右眼关键点索引，左眼36~41，右眼42~47
(lStart,lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart,rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# 获取摄像头数据
vs = VideoStream(src=0).start()
fileStream = False
time.sleep(1.0)

while True:
    # 获取数据帧,调整大小并进行灰度化
    frame = vs.read()
    frame = imutils.resize(frame,width=660)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # 得到脸的位置,获取人脸图像rects
    rects = detector(gray,0)
    # 逐张人脸进行特征点提取
    for rects in rects:
        # 将脸部关键点信息转化成numpy数组
        shape = predictor(gray,rects)
        shape = face_utils.shape_to_np((shape))

        # 分别计算左右眼的EAR
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # 求平均值
        ear = (leftEAR + rightEAR) / 2.0

        # 分别计算左右眼的包
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        # 画眼部轮廓
        cv2.drawContours(frame,[leftEyeHull],-1,(0,255,0),1)
        cv2.drawContours(frame,[rightEyeHull],-1,(0,255,0),1)

        if ear < 0.3:
            COUNTER += 1
        else:
            if COUNTER >= 3 :
                TOTAL += 1

            # 重置计数器
            COUNTER = 0



       # 显示眨眼的次数和EAR
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

   # 显示图像
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    # 按q键，退出循环
    if key == ord("q"):
        break

# 资源清理
cv2.destroyAllWindows()
vs.stop()






