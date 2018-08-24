import cv2
import sys
import os

"""--------读取已有的视频并截取图片--------"""
# 配置相关参数
video_path=sys.argv[1]              # 视频文件路径
save_path=sys.argv[2]               # 截图保存路径
num_frame=int(sys.argv[3])          # 要截取图片数
frame_interval=int(sys.argv[4])     # 帧采样间隔
init_frame=int(sys.argv[5])         # 起始采样帧

print("video_path:",video_path)
print("save_path:",save_path)
print("num_frame:",num_frame)
print("frame_interval:",frame_interval)
print("init_frame:",init_frame)

# 判断路径是否存在
if not os.path.exists(video_path):
    print("video path {} doesn't exists".format(video_path))
    exit(0)

if not os.path.exists(save_path):
    print("create save path {}".format(save_path))
    os.mkdir(save_path)

# 定义一个videocapture
cap=cv2.VideoCapture()

# 打开视频
cap.open(video_path)
print("video info:")
print('\ttotal frames:',int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
print('\tframe rate:',int(cap.get(cv2.CAP_PROP_FPS)))
print('\tframe size:',(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# 获取视频总帧数
total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 避开开头若干帧帧
for i in range(init_frame):
    cap.read()

# 对剩余的帧进行截取图片
count=0
for i in range(total_frames-init_frame):
    _,frame=cap.read()

    if i%frame_interval==0:
        img_name='frame_{:0>5d}.jpg'.format(int(i/frame_interval+1))
        img_path=os.sep.join([save_path,img_name])
        frame=cv2.resize(frame,(1920,1200))
        cv2.imwrite(img_path,frame)
        count+=1

    # 每隔50张打印一下进度
    if ((i/frame_interval+1)%50==0):
        print('generated {} pictures....'.format(count))

    # 如果达到总张数就停止
    if count==num_frame:
        print("successfully generated {} pictures".format(num_frame))
        break

# 若总张数每到，就打印实际产生的张数
if count<num_frame:
    print("practically generated {} pictures".format(count))

