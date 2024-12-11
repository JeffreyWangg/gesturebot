from mmdet.apis import init_detector, inference_detector, DetInferencer

config_file = 'rtmdet_tiny_8xb32-300e_coco.py'
checkpoint_file = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
inferencer = DetInferencer(config_file, checkpoint_file, device="cpu")

# Perform inference
output = inferencer('couch2.jpeg')

def non_max_suppression(boxes, scores, threshold):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        for j in order:
            # Calculate the IoU between the two boxes
            intersection = max(0, min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0])) * \
                           max(0, min(boxes[i][3], boxes[j][3]) - max(boxes[i][1], boxes[j][1]))
            union = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) + \
                    (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]) - intersection
            iou = intersection / union

            if iou > threshold:
                order.remove(j)
    return keep

keep = non_max_suppression(output['predictions'][0]['bboxes'], output['predictions'][0]['scores'], 0.5)
print(keep)
# print([output['predictions'][0]['labels'][i] for i in keep], [output['predictions'][0]['scores'][i] for i in keep])
# print([output['predictions'][0]['bboxes'][i] for i in keep])
# ok this works

# def sort_arrays(array1, array2):
#     paired = sorted(zip(array1, array2), key=lambda x: x[0], reverse=True)
#     sorted_array1, sorted_array2 = zip(*paired)
#     return list(sorted_array1), list(sorted_array2)

# scores, labels = sort_arrays(output['predictions'][0]['scores'], output['predictions'][0]['labels'])
# print(labels, scores)

