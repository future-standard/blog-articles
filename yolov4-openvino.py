#!/usr/bin/env python3

import sys
import cv2
import numpy
import random
from math import exp
from openvino.inference_engine import IENetwork, IECore


def calc_iou(a_, b_):

    # Find the intersection: a_ & b_
    x0 = max(a_['x0'], b_['x0'])
    x1 = min(a_['x1'], b_['x1'])
    y0 = max(a_['y0'], b_['y0'])
    y1 = min(a_['y1'], b_['y1'])

    if x0 < x1 and y0 < y1:
        i_area = (x1 - x0) * (y1 - y0)
    else:
        return 0.0

    # Find the union: a_ + b_
    a_area = (a_['x1'] - a_['x0']) * (a_['y1'] - a_['y0'])
    b_area = (b_['x1'] - b_['x0']) * (b_['y1'] - b_['y0'])
    u_area = a_area + b_area - i_area

    return i_area / u_area



def apply_nms(objs_):

    # First choose the most confident bounding boxes as the true detections
    # Then, filter out the other boxes if they have high IoU to them

    objs_.sort(key=lambda x: x['confidence'], reverse=True)

    survivors = []
    for i, obj in enumerate(objs_):
        if obj['kind'] is None:
            continue            # Crossed out by higher confident object
        else:
            survivors.append(obj)

        for victim in objs_[i + 1:]:
            if victim['kind'] is None:
                continue        # Already dead; ignore it

            iou = calc_iou(obj, victim)

            if victim['kind'] == obj['kind']:
                if iou > nms:
                    victim['kind'] = None

    return survivors



def entry_index(side_sq_, coord_, classes_, location_, entry_):
    n   = location_ // side_sq_
    loc = location_ % side_sq_
    return int(side_sq_ * (n * (coord_ + classes_ + 1) + entry_) + loc)



def parse_yolo_region(blob_, anchors_, h_, w_, img_h_, img_w_, min_confidence_):

    # Sanity check on blob_
    _, _, blob_h, blob_w = blob_.shape
    assert blob_h == blob_w, f"Blob's hight and width must be same: {blob_.shape}"

    predict = blob_.flatten()
    side    = blob_h
    side_sq = side * side

    objs = []
    for i in range(side_sq):
        row = i // side
        col = i % side

        for n in range(yolo_anchor_count):
            location = n * side_sq + i
            obj_idx = entry_index(side_sq, yolo_coords, len(labels), location, yolo_coords)

            scale = 1. / (exp(-predict[obj_idx]) + 1)
            if scale < min_confidence_:
                continue

            box_idx = entry_index(side_sq, yolo_coords, len(labels), location, 0)

            x = 1. / (exp(-predict[box_idx + 0 * side_sq]) + 1)
            y = 1. / (exp(-predict[box_idx + 1 * side_sq]) + 1)

            # Network produces location predictions in absolute coordinates of feature maps.
            # Scale it to relative coordinates.
            x = (col + x) / side
            y = (row + y) / side

            # Normalize sizes by input shape
            # Note that exp() can become a very big number. Hence, try & except.
            try:
                w = exp(predict[box_idx + 2 * side_sq]) * anchors_[2 * n] / w_
                h = exp(predict[box_idx + 3 * side_sq]) * anchors_[2 * n + 1] / h_
            except OverflowError:
                continue

            for k in range(len(labels)):
                class_idx = entry_index(side_sq, yolo_coords, len(labels), location, yolo_coords + 1 + k)

                confidence = scale * 1. / (exp(-predict[class_idx]) + 1)
                if confidence < min_confidence_:
                    continue

                if k < len(labels):
                    label = labels[k]
                else:
                    label = f'({k})'

                obj = {
                    'kind': label,
                    'confidence': confidence,
                    'x0': int(numpy.clip(round((x - w / 2) * img_w_), 0, img_w_ - 1)),
                    'y0': int(numpy.clip(round((y - h / 2) * img_h_), 0, img_h_ - 1)),
                    'x1': int(numpy.clip(round((x + w / 2) * img_w_), 0, img_w_ - 1)),
                    'y1': int(numpy.clip(round((y + h / 2) * img_h_), 0, img_h_ - 1))
                }

                objs.append(obj)

    return objs




# ---- Program starts here ----


# Load the category labels
with open('coco.names', 'r') as f:
    labels = list(map(str.strip, f.readlines()))
assert(len(labels) > 0)

label_colors = {}
for label in labels:
    label_colors[label] = (random.randint(50, 250), random.randint(50, 250), random.randint(50, 250))


# Load the model data to OpenVINO
ie                  = IECore()
net                 = ie.read_network(model='yolov4.xml', weights='yolov4.bin')
input_layer         = next(iter(net.input_info))    # The first key of the dictionary: input_info
N, C, net_h, net_w  = net.input_info[input_layer].input_data.shape

net.batch_size      = 1
exec_net            = ie.load_network(network=net, device_name='CPU', num_requests=1)

nms                 = 0.6
min_confidence      = 0.7

print(f'Input Layers            : {list(net.input_info)}')
print(f'Output Layers           : {list(net.outputs.keys())}')
print(f'NMS IoU threshold       : {nms * 100} %')
print(f'Confidence threshold    : {nms * 100} %')
print(f'\n-------- Class Labels --------\n{labels}\n------------------------------\n')


# The parameters dependent on YOLO algorithm type
yolo_coords         = 4
yolo_anchor_count   = 3
yolo_anchors        = {
    "conv2d_93/BiasAdd/Add" : [12,16, 19,36, 40,28],
    "conv2d_101/BiasAdd/Add": [36,75, 76,55, 72,146],
    "conv2d_109/BiasAdd/Add": [142,110, 192,243, 459,401],
}


# Load the target image
assert(sys.argv[1])
img = cv2.imread(sys.argv[1])
img_h, img_w = img.shape[:2]


# Resize the image to the network size
if net_w > img_w or net_h > img_h:
    resized = cv2.resize(img, (net_w, net_h), interpolation=cv2.INTER_LINEAR)
else:
    resized = cv2.resize(img, (net_w, net_h), interpolation=cv2.INTER_AREA)


# Convert data layout from HWC to CHW: Axis [y,x,bgr] -> [bgr,y,x]
resized = resized.transpose((2, 0, 1))
resized = resized.reshape((N, C, net_h, net_w))


# Perform an inference
exec_net.requests[0].infer({input_layer: resized})
outputs = exec_net.requests[0].output_blobs


# Extract the detected objects
objs = []
for output_layer, blob in outputs.items():
    blob.buffer.reshape(net.outputs[output_layer].shape)
    objs += parse_yolo_region(blob.buffer, yolo_anchors[output_layer], net_h, net_w, img_h, img_w, min_confidence)
objs = apply_nms(objs)


# Draw the bounding boxes, categories, and confidences
for obj in objs:
    x0, y0, x1, y1 = obj['x0'], obj['y0'], obj['x1'], obj['y1']
    kind, confidence = obj['kind'], obj['confidence']

    cv2.rectangle(img, (x0, y0), (x1, y1), label_colors[kind], 3)
    cv2.putText(img, f'{kind} {confidence:.3f}',
                (x0 + 5, y0 - 8), cv2.FONT_HERSHEY_DUPLEX, 1, label_colors[kind])


# Dump the result image
written = cv2.imwrite('result.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
if written is True:
    print('done', flush=True)
else:
    print('failed', flush=True)


# EOF
