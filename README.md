# WORKSHOP--4-Coin-Detection-using-OpenCV
## Aim:
To detect and count the number of coins in an image using OpenCV by applying grayscale conversion, thresholding, morphological operations, blob detection, and contour detection.

## Algorithm:
1.Import the required OpenCV, NumPy, and Matplotlib libraries. 
2.Read and display the input coin image. 
3.Convert the input image from BGR to grayscale.
4.Apply thresholding to separate the coins from the background. 
5.Perform morphological Opening to remove small noise. 6.Perform morphological Closing to fill small gaps and improve the coin regions. 
7.Apply Blob Detection to detect coin-like objects and count them. 
8.Apply Contour Detection to identify the boundaries of the coins and count them. 
9.Display the detected coins and their boundaries.
10.Compare the results obtained using Blob Detection and Contour Detection.

## Program
  ```
Original Image:
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("coin.png")

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")
plt.show()
```
<img width="533" height="571" alt="image" src="https://github.com/user-attachments/assets/7e43660c-5a64-4824-87e3-3c6a0431585d" />

```
Grayscale Image:
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(8, 6))
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()
```
<img width="542" height="562" alt="image" src="https://github.com/user-attachments/assets/9b4c10cb-9907-4f77-ba43-0bccee9b940b" />

```
Thresholding:
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(8, 6))
plt.imshow(thresh, cmap="gray")
plt.title("Thresholded Image")
plt.axis("off")
plt.show()
```
<img width="550" height="573" alt="image" src="https://github.com/user-attachments/assets/98aba016-5da1-4a21-9ee3-79edd405e860" />

```
Morphological Opening & Closing
kernel = np.ones((5, 5), np.uint8)

# Opening - removes small noise
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Closing - fills small holes
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(opening, cmap="gray")
plt.title("After Opening")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(closing, cmap="gray")
plt.title("After Closing")
plt.axis("off")

plt.show()
```
<img width="912" height="440" alt="image" src="https://github.com/user-attachments/assets/40335840-e115-4694-955b-a738bf376548" />


```
Blob Detection:
params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 500
params.maxArea = 50000

params.filterByCircularity = True
params.minCircularity = 0.7

params.filterByConvexity = True
params.minConvexity = 0.8

params.filterByInertia = True
params.minInertiaRatio = 0.5

detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(closing)

blob_image = cv2.drawKeypoints(
    cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR),
    keypoints,
    None,
    (0, 0, 255),
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(blob_image, cv2.COLOR_BGR2RGB))
plt.title("Blob Detection")
plt.axis("off")
plt.show()

print("Number of coins detected using Blob Detection:", len(keypoints))
```
<img width="560" height="585" alt="image" src="https://github.com/user-attachments/assets/2d3b8e32-6017-45d5-8933-02aa83cc1037" />


```
Contour Detection
contours, hierarchy = cv2.findContours(
    closing,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Filter small contours
coin_contours = []

for contour in contours:
    area = cv2.contourArea(contour)
    
    if area > 500:
        coin_contours.append(contour)

# Draw contours
contour_image = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)

cv2.drawContours(
    contour_image,
    coin_contours,
    -1,
    (0, 255, 0),
    2
)

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.title("Contour Detection")
plt.axis("off")
plt.show()

print("Number of coins detected using Contours:", len(coin_contours))
Coin counting
print("===== RESULTS =====")
print("Coins detected using Blob Detection    :", len(keypoints))
print("Coins detected using Contour Detection :", len(coin_contours))
```
<img width="567" height="597" alt="image" src="https://github.com/user-attachments/assets/9481e1d9-1a12-4e07-92df-5e591ea09ed2" />


Discussion and conclusion:
## Result:

The coins in the input image were successfully detected and counted using both Blob Detection and Contour Detection. The thresholding and morphological operations helped to improve the image quality and separate the coins from the background.

The number of coins detected using Blob Detection was obtained from the detected keypoints, while the number of coins detected using Contour Detection was obtained from the filtered contours.
