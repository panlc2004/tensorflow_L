import math

from src.mtcnn.mtcnn import MTCNN
import cv2

image = cv2.imread("052.jpg")
detector = MTCNN()
result = detector.detect_faces(image)
print(result)

bounding_box = result[0]['box']
keypoints = result[0]['keypoints']

cv2.rectangle(image,
              (bounding_box[0], bounding_box[1]),
              (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
              (0, 155, 255),
              2)

cv2.circle(image, (keypoints['left_eye']), 2, (0, 155, 255), 2)
cv2.circle(image, (keypoints['right_eye']), 2, (0, 155, 255), 2)
cv2.circle(image, (keypoints['nose']), 2, (0, 155, 255), 2)
cv2.circle(image, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
cv2.circle(image, (keypoints['mouth_right']), 2, (0, 155, 255), 2)

center = ((keypoints['left_eye'][0] + keypoints['right_eye'][0] + keypoints['mouth_left'][0] + keypoints['mouth_right'][0]) // 4,
          (keypoints['left_eye'][1] + keypoints['right_eye'][1] + keypoints['mouth_left'][1] + keypoints['mouth_right'][1]) // 4)
cv2.circle(image, center, 2, (0, 255, 255), 2)

cv2.imwrite("ivan_drawn.jpg", image)


def AlignWuXiang(input_image, keypoints, output_size=(96, 112), ec_mc_y=40):
    eye_center = ((keypoints['left_eye'][0] + keypoints['right_eye'][0]) / 2,
                  (keypoints['left_eye'][1] + keypoints['right_eye'][1]) / 2)
    mouth_center = ((keypoints['mouth_left'][0] + keypoints['mouth_right'][0]) / 2, (keypoints['mouth_left'][1] + keypoints['mouth_right'][1]) / 2)
    angle = math.atan2(mouth_center[0] - eye_center[0], mouth_center[1] - eye_center[1]) / math.pi * -180.0
    # angle = math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0]) / math.pi * 180.0
    scale = ec_mc_y / math.sqrt((mouth_center[0] - eye_center[0]) ** 2 + (mouth_center[1] - eye_center[1]) ** 2)
    center = ((keypoints['left_eye'][0] + keypoints['right_eye'][0] + keypoints['mouth_left'][0] + keypoints['mouth_right'][0]) / 4,
              (keypoints['left_eye'][1] + keypoints['right_eye'][1] + keypoints['mouth_left'][1] + keypoints['mouth_right'][1]) / 4)
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
    rot_mat[0][2] -= (center[0] - output_size[0] / 2)
    rot_mat[1][2] -= (center[1] - output_size[1] / 2)
    warp_dst = cv2.warpAffine(input_image, rot_mat, output_size)
    return warp_dst


alignedImg = AlignWuXiang(image, keypoints)
cv2.imwrite("test-aligned.jpg", alignedImg)



import tensorflow as tf
tf.train.GradientDescentOptimizer