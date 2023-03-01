# https://debuggercafe.com/object-detection-using-pytorch-faster-rcnn-resnet50-fpn-v2/#download-code
coco_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


import torchvision.transforms as transforms
import cv2
import numpy as np
import torch
np.random.seed(42)
# Create different colors for each class.
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))
# Define the torchvision image transforms.
transform = transforms.Compose([
    transforms.ToTensor(),
])
def predict(image, model, device, detection_threshold):
    """
    Predict the output of an image after forward pass through
    the model and return the bounding boxes, class names, and
    class labels.
    """
    # Transform the image to tensor.
    image = transform(image).to(device)
    # Add a batch dimension.
    image = image.unsqueeze(0)
    # Get the predictions on the image.
    with torch.no_grad():
        outputs = model(image)
    # Get score for all the predicted objects.
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # Get all the predicted bounding boxes.
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # Get boxes above the threshold score.
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    labels = outputs[0]['labels'][:len(boxes)]
    # Get all the predicited class names.
    pred_classes = [coco_names[i] for i in labels.cpu().numpy()]
    return boxes, pred_classes, labels

def draw_boxes(boxes, classes, labels, image):
    """
    Draws the bounding box around a detected object.
    """
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)  # Line width.
    tf = max(lw - 1, 1) # Font thickness.
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            img=image,
            pt1=(int(box[0]), int(box[1])),
            pt2=(int(box[2]), int(box[3])),
            color=color[::-1],
            thickness=lw
        )
        cv2.putText(
            img=image,
            text=classes[i],
            org=(int(box[0]), int(box[1]-5)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=lw / 3,
            color=color[::-1],
            thickness=tf,
            lineType=cv2.LINE_AA
        )
    return image

import torchvision
def get_model(device='cpu', model_name='v2'):
    # Load the model.
    if model_name == 'v2':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights='DEFAULT'
        )
    elif model_name == 'v1':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights='DEFAULT'
        )
    # Load the model onto the computation device.
    model = model.eval().to(device)
    return model

import torch
import argparse
import cv2
import numpy as np
from PIL import Image
# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', default='input/image_1.jpg',
    help='path to input input image'
)
parser.add_argument(
    '-t', '--threshold', default=0.5, type=float,
    help='detection threshold'
)
parser.add_argument(
    '-m', '--model', default='v2',
    help='faster rcnn resnet50 fpn or fpn v2',
    choices=['v1', 'v2']
)
args = vars(parser.parse_args())


# Define the computation device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(device, args['model'])
# Read the image.
# image = Image.open(args['input']).convert('RGB')
image = Image.open('home.jpg').convert('RGB')
# Create a BGR copy of the image for annotation.
image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
# Detect outputs.
with torch.no_grad():
    boxes, classes, labels = predict(image, model, device, args['threshold'])
# Draw bounding boxes.
image = draw_boxes(boxes, classes, labels, image_bgr)
save_name = f"{args['input'].split('/')[-1].split('.')[0]}_t{''.join(str(args['threshold']).split('.'))}_{args['model']}"
cv2.imshow('Image', image)
cv2.imwrite(f"outputs/{save_name}.jpg", image)
cv2.waitKey(0)

# import urllib.request
#
# # URL of the image to be downloaded
# url = 'https://example.com/image.jpg'
# url = 'data:image/jpeg;base64,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'
#
# # Local path to save the image
# filename = 'home.jpg'
# filename = 'image.jpg'
#
# # Download the image from the URL and save it to the local path
# urllib.request.urlretrieve(url, filename)
#
# import torch
# import cv2
# from matplotlib import pyplot as plt
#
# # Load the model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#
# # Enable Autoshape.
# # model = model.summary()
#
# # Print the model summary.
# print(model.names)
#
# # Load the image
# img = cv2.imread(filename)
#
#
# # Convert the image to RGB
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# # Detect objects in the image
# results = model(img, size=640)
#
# # Get the bounding box coordinates, labels, and scores for each object detected
# boxes = results.xyxy[0].cpu().numpy()
# class_names = results.names
# # Get the unique class IDs from the results
# class_ids = set(results.xyxy[0][:, -1].long().cpu().numpy())
# if len(results.xyxy[0]) > 0:
#     labels = labels = tuple([class_names[id] for id in class_ids])
# else:
#     labels = []
# scores = results.xyxy[0][:, -1].cpu().numpy()
#
# # Draw the bounding boxes on the image
# for box, label, score in zip(boxes, labels, scores):
#     if score > 0.5:
#         img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
#         img = cv2.putText(img, f'{label} {score:.2f}', (int(box[0]), int(box[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
# # Show the resulting image
# # plt.imshow(img)
# # Display the results
# plt.imshow(cv2.cvtColor(results.render()[0][..., ::-1], cv2.COLOR_BGR2RGB))
# results.show()
# plt.show()
#

