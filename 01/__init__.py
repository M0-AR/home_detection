# Works
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
#
# # Load pre-trained ResNet50 model
# model = tf.keras.applications.ResNet50(
#     include_top=True,
#     weights='imagenet',
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=1000,
#     classifier_activation='softmax'
# )
#
# # Load image and preprocess it
# img_path = 'home.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
#
# # Make prediction
# preds = model.predict(x)
# pred_class = decode_predictions(preds, top=1)[0][0][1]
#
# # Print prediction
# if pred_class == 'murermestervillaen fra 50erne':
#     print('The image contains Murermestervillaen fra 50erne architecture style')
# else:
#     print('The image does not contain Murermestervillaen fra 50erne architecture style')
#
# print(pred_class)

# Works
# from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
# from PIL import Image
# import numpy as np
#
# # Load pre-trained VGG16 model and its weights
# model = VGG16(weights='imagenet', include_top=True)
#
# # Load image to be classified
# img_path = 'home.jpg'
# img = Image.open(img_path).resize((224, 224)) # VGG16 input size
#
# # Preprocess image for input to VGG16 model
# x = np.array(img).astype(np.float32)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
#
# # Pass image through VGG16 model to obtain feature vector
# features = model.predict(x)
#
# # Classify image based on feature vector
# prediction = decode_predictions(features, top=1)[0][0]
# print('Predicted class:', prediction[1])

# ---------------------------------------------------------------------------
# Works: Roof with large overhang and 45-degree slope:
# One way to detect a roof with a large overhang and 45-degree
# slope is to use edge detection algorithms like Canny or Sobel to extract the edges of the image, then use Hough
# transforms to identify lines and shapes in the image. Look for prominent lines that form a 45-degree angle with the
# horizontal, and check if they are part of the roof structure. You can also check if the roof has a large overhang
# by looking for lines that extend beyond the walls of the house.
# import cv2
# import numpy as np
#
# # Load image
# img = cv2.imread('home.jpg')
#
# # Convert to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # Detect prominent roof with large overhang
# edges = cv2.Canny(gray, 50, 150)
# lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=5)
#
# # Detect 45-degree roof slope
# sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
# sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
# theta = np.arctan2(sobely, sobelx)
# theta = np.abs(theta * 180 / np.pi)
#
# # Check if any lines satisfy the roof criteria
# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     if abs(y2-y1) > abs(x2-x1) and 40 <= theta[y1, x1] <= 50:
#         cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#
# # Display result
# cv2.imshow('Result', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ---------------------------------------------------------------------------
# Find the roof and outside edges
# import cv2
# import numpy as np
#
# # Load image
# from matplotlib import pyplot as plt
#
# img = cv2.imread('home_01.jpg')
#
# # Convert to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # Apply Gaussian blur to reduce noise
# blur = cv2.GaussianBlur(gray, (3, 3), 0)
#
# # Apply Canny edge detection
# edges = cv2.Canny(blur, 100, 200)
#
# # Apply Hough transform to detect lines
# lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=5)
#
# # Draw lines on original image
# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#
# # Resize the image to half its size
# resized_image = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
#
#
# # Show result
# cv2.imshow('Brick Exterior with Timber Framing', resized_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ---------------------------------------------------------------------------
# by color
# import cv2
# import numpy as np
#
# # Load image
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
#
# img = cv2.imread('home_03.jpg')
#
# # Convert to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # Apply thresholding to segment the brick walls
# _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
#
# # Find contours of the brick walls
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # Draw rectangles around the brick wall contours
# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
# # Show the image with brick wall contours marked by rectangles
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()

# # Get color of the brick walls by computing the mean color of the pixels in the contours
# brick_color = np.mean(img[contours[0][:, 0][:, 1], contours[0][:, 0][:, 0]], axis=0)
#
# print(brick_color)  # Output: [167.83333333  77.16666667  54.66666667]
#
# import webcolors
#
# color = (brick_color[0], brick_color[1], brick_color[2])
#
# # Round the color values to integers
# color = tuple(round(c) for c in color)
# print(color)
#
# hex_string = "#{:02X}{:02X}{:02X}".format(*color)
# print(hex_string)  # Output: '#345680'
#
#
# # Get the closest color name for the RGB values using the webcolors module
#
# def get_color_name(rgb_color):
#     red, green, blue = rgb_color
#
#     if red > 200 and green < 50 and blue < 50:
#         return "red"
#     elif red < 50 and green > 200 and blue < 50:
#         return "green"
#     elif red < 50 and green < 50 and blue > 200:
#         return "blue"
#     elif red > 200 and green > 200 and blue < 50:
#         return "yellow"
#     elif red > 200 and green < 50 and blue > 200:
#         return "magenta"
#     elif red < 50 and green > 200 and blue > 200:
#         return "cyan"
#     else:
#         return "unknown"
#
#
# color = (52, 86, 128)
# print(get_color_name(color))

# # Create a 1x1 image with the specified color
# im = np.zeros((1, 1, 3), dtype=np.uint8)
# im[:, :, 0] = brick_color[0]
# im[:, :, 1] = brick_color[1]
# im[:, :, 2] = brick_color[2]
#
# # Display the color patch using matplotlib
# fig, ax = plt.subplots()
# ax.imshow(im)
# plt.show()
# ---------------------------------------------------------------
# By roof
# import cv2
# import numpy as np
#
# # Load image
# img = cv2.imread('home.jpg')
#
# # Convert to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # Apply Canny edge detection to find edges in the image
# edges = cv2.Canny(gray, 100, 200)
#
# # Apply Hough transform to detect lines in the image
# lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
#
# # Loop over the lines and analyze their characteristics
# for line in lines:
#     rho, theta = line[0]
#     if theta > np.pi/4 and theta < 3*np.pi/4:
#         print("Typehuset")
#     else:
#         print("muremestervillaen")
# ---------------------------------------------------------
# The most important code line on roofs
# import cv2
# import numpy as np
#
# # Load image
# img = cv2.imread('home_03.jpg')
#
# # Convert to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # Apply Gaussian blur to reduce noise
# blur = cv2.GaussianBlur(gray, (3, 3), 0)
#
# # Apply Canny edge detection
# edges = cv2.Canny(blur, 100, 200)
#
# # Apply Hough transform to detect lines
# lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=5)
#
# # Check if the image contains a low brick facade with a separate roof structure
# brick_facade = False
# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     if y1 == y2 and abs(x1 - x2) > 200:
#         brick_facade = True
#         break
#
# # Check if the image has a prominent roof with visible rafters and a large overhang
# prominent_roof = False
# if not brick_facade:
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         if abs(y1 - y2) > 100 and abs(x1 - x2) < 20:
#             prominent_roof = True
#             break
#
# # Determine the type of house based on the detected features
# if brick_facade:
#     print("Typehuset")
# else:
#     print("muremestervillaen")
#
# # Draw lines on original image
# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#
# # # Resize the image to half its size
# resized_image = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
#
# # Show result
# cv2.imshow('Brick Exterior with Timber Framing', resized_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# -------------------------------------------------------------------------------------

# Here's the combined code to find windows and display an image with the window marking:
# Load image
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

img = cv2.imread('home_01.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to segment the brick walls
_, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

# Find contours of the brick walls
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw rectangles around the brick wall contours
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Find and draw rectangles around the window contours within the brick wall contour
    roi = gray[y:y+h, x:x+w]
    edges = cv2.Canny(roi, 50, 150)
    window_contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for window_contour in window_contours:
        window_x, window_y, window_w, window_h = cv2.boundingRect(window_contour)
        cv2.rectangle(img, (x + window_x, y + window_y), (x + window_x + window_w, y + window_y + window_h), (0, 0, 255), 2)

# Show the image with brick wall and window contours marked by rectangles
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
