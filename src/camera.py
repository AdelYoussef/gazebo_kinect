#!/usr/bin/env python

import rospy
import cv2
import tf
import ros_numpy
import numpy as np 
import imutils 
import message_filters
from std_msgs.msg import Int32 , Float64MultiArray , Int32MultiArray
from cv_bridge import CvBridge
from sensor_msgs.msg import Image,CameraInfo , PointCloud2
from geometry_msgs.msg import PointStamped,PoseStamped


rospy.init_node('camera.py', anonymous=True)
pub=rospy.Publisher('centers_rgb',Int32MultiArray,queue_size = 10)
rate = rospy.Rate(0.5)
pt = PointStamped()
t = tf.TransformListener()
pos = Int32MultiArray()

def centers(hsv,low,high,cv_image,frame):
    mask = cv2.inRange(hsv, low, high)
    cx ,cy = 0,0
    drawContours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        area = cv2.contourArea(c)
        if area> 5:
            
            M = cv2.moments(c)
            
            cx = int(M["m10"]/ M["m00"])
            cy = int(M["m01"]/ M["m00"])
            ccx=str(cx)
            ccy=str(cy)

            cv2.drawContours(frame, [c], -1, (0,255,0), 1)
            cv2.circle(frame, (cx, cy), 7, (255,255,255), -1)
            cv2.putText(frame,ccx,(cx+20, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),2)
            cv2.putText(frame,ccy,(cx+60, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),2)
    
    z = cv_image[cx,cy]
   

    x = (cx - 399.711466) * z / 476.470489
    y = (cy - 399.785066) * z / 476.362329
    x,y,z = x+0.4 , -y, 2-z+0.2
    x = np.round(x,1)
    y = np.round(y,1)
    
    #cv2.imshow("Frame", frame)
    #cv2.imshow("mask",mask)
    #cv2.waitKey(3)
    
    return(x,y)
def callback(data,depth,pc):

    
    env = np.zeros((4,2),dtype=float)
    cv_image = CvBridge().imgmsg_to_cv2(depth, "passthrough")
    depth_array = np.array(cv_image, dtype=np.float64)
    cv_image_norm = cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)


    frame = CvBridge().imgmsg_to_cv2(data, "bgr8")
    


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_blue=np.array([70,0,0])
    high_blue=np.array([255,255,255])

    low_red=np.array([0,0,61])
    high_red=np.array([0,255,255])

    low_green=np.array([40,0,0])
    high_green=np.array([100,255,255])

    low_yellow=np.array([20,0,0])
    high_yellow=np.array([55,255,255])
    
    center_b = centers(hsv, low_blue, high_blue,cv_image,frame)
    center_r = centers(hsv, low_red, high_red,cv_image,frame)
    center_g = centers(hsv, low_green, high_green,cv_image,frame)
    center_y = centers(hsv, low_yellow, high_yellow,cv_image,frame)
    env[0,:]=center_b
    env[1,:]=center_r
    env[2,:]=center_g
    env[3,:]=center_y
    pos.data = env
    pub.publish(pos)



    #xyz_array = ros_numpy.point_cloud2.get_xyz_points(pc)
   # pc_img = CvBridge().imgmsg_to_cv2()

    rospy.loginfo("blue")
    rospy.loginfo(center_b)
    rospy.loginfo("red")
    rospy.loginfo(center_r)
    rospy.loginfo("green")
    rospy.loginfo(center_g)
    rospy.loginfo("yellow")
    rospy.loginfo(center_y)
   # cv2.waitKey(3)
       
            
image_sub = message_filters.Subscriber('/abb_irb120_3_58/camera1/image_raw', Image)           
depth_sub = message_filters.Subscriber("/abb_irb120_3_58/camera1/depth_raw", Image)
point_cloud_sub = message_filters.Subscriber("/abb_irb120_3_58/camera1/point_cloud", PointCloud2)
#label = message_filters.Subscriber('BCI_Command',Int32)           

ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub,point_cloud_sub], queue_size=5, slop=0.1)
ts.registerCallback(callback)
rospy.spin()


    