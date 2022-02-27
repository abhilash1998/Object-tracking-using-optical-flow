import numpy as np
import cv2

cap = cv2.VideoCapture("Cars on Highway.mp4")
_, img = cap.read()
img = cv2.resize(img, (640,480))
prev = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(img)
hsv[..., 1] = 255
#out = cv2.VideoWriter('Answer1_1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (640,480))
#out = cv2.VideoWriter('Answer1_2_1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (640,480))
out = cv2.VideoWriter('Answer1_2_2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (640,480))

while True:
    r, frame = cap.read()
    if frame is None:
        break
    frame = cv2.resize(frame, (640,480))
    next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_copy = frame.copy()
    fl = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flCopy = fl.copy()
    m, theta = cv2.cartToPolar(fl[:, :, 0], fl[:, :, 1])
    m = m * 4
    print(frame.shape)
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if i % 25 == 0 and j % 25 == 0:
                tip = (j, i)
                end = (
                i + (m[i, j] * np.sin(theta[i, j])), j + (m[i, j] * np.cos(theta[i, j])))
                frame_copy = cv2.arrowedLine(frame_copy, tip, (int(end[1]), int(end[0])), (0, 255, 255), 2)
    frame2 = np.zeros_like(frame)
    _ = flCopy[:, :, 0]
    _ = 2.5 * _
    arr = np.where(_ > 1)
    for i in range(len(arr[0])):
        frame2[arr[0][i], arr[1][i], :] = frame[arr[0][i], arr[1][i], :]
    hsv[..., 0] = theta * (180 / (np.pi / 2))
    hsv[..., 2] = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('BGR', bgr)
    cv2.imshow('Vectors', frame_copy)
    cv2.imshow('Vehicles', frame2)
    #out.write(frame_copy)
    #out.write(bgr)
    out.write(frame2)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    prev = next

out.release()
cv2.destroyAllWindows()
