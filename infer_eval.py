from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing import image
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd512 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from data_generator.object_detection_2d_data_generator import DataGenerator
from eval_utils.average_precision_evaluator import Evaluator

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

# Set a few configuration parameters.
img_height = 512
img_width = 512
n_classes = 3
model_mode = 'training'  # Or training

# TODO: Set the paths to the dataset here.
dataset_images_dir = ['/Users/justinbutler/Desktop/test/tiny_test/M0101/JPEGImages']
dataset_annotations_dir = ['/Users/justinbutler/Desktop/test/tiny_test/M0101/Annotations/']
dataset_image_set_filename = ['/Users/justinbutler/Desktop/test/tiny_test/M0101/img_set.txt']

# For image for inference
single_img_path = '/Users/justinbutler/Desktop/test/extra_tiny_test/M0201/JPEGImages/img000001.jpg'

# Load model
# TODO: Set the path to the `.h5` file of the model to be loaded.
model_path = '/Users/justinbutler/Desktop/school/Calgary/ML_Work/ssd_keras/test_trained.h5'

# The XML parser needs to now what object class names to look for
# and in which order to map them to integers.
classes = ['background',
           'bus', 'car', 'truck']

# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

K.clear_session()  # Clear previous models from memory.

model_objects = {'AnchorBoxes': AnchorBoxes,
                 'L2Normalization': L2Normalization,
                 'DecodeDetections': DecodeDetections,
                 'compute_loss': ssd_loss.compute_loss}

model = load_model(model_path, custom_objects=model_objects)

dataset = DataGenerator()

dataset.parse_xml(images_dirs=dataset_images_dir,
                  image_set_filenames=dataset_image_set_filename,
                  annotations_dirs=dataset_annotations_dir,
                  classes=classes,
                  include_classes='all',
                  exclude_truncated=False,
                  exclude_difficult=False,
                  ret=False)

# Evaluate Average Precision
evaluator = Evaluator(model=model,
                      n_classes=n_classes,
                      data_generator=dataset,
                      model_mode=model_mode)

results = evaluator(img_height=img_height,
                    img_width=img_width,
                    batch_size=8,
                    data_generator_mode='resize',
                    round_confidences=False,
                    matching_iou_threshold=0.5,
                    border_pixels='include',
                    sorting_algorithm='quicksort',
                    average_precision_mode='sample',
                    num_recall_points=11,
                    ignore_neutral_boxes=True,
                    return_precisions=True,
                    return_recalls=True,
                    return_average_precisions=True,
                    verbose=True)

mean_average_precision, average_precisions, precisions, recalls = results

# Display results
for i in range(1, len(average_precisions)):
    print("{:<14}{:<6}{}".format(
        classes[i], 'AP', round(average_precisions[i], 3)))
print()
print("{:<14}{:<6}{}".format('', 'mAP', round(mean_average_precision, 3)))

m = max((n_classes + 1) // 2, 2)
n = 2

fig, cells = plt.subplots(m, n, figsize=(n*8, m*8))
for i in range(m):
    for j in range(n):
        if n*i+j+1 > n_classes:
            break
        cells[i, j].plot(recalls[n*i+j+1], precisions[n*i+j+1],
                         color='blue', linewidth=1.0)
        cells[i, j].set_xlabel('recall', fontsize=14)
        cells[i, j].set_ylabel('precision', fontsize=14)
        cells[i, j].grid(True)
        cells[i, j].set_xticks(np.linspace(0, 1, 11))
        cells[i, j].set_yticks(np.linspace(0, 1, 11))
        cells[i, j].set_title("{}, AP: {:.3f}".format(
            classes[n*i+j+1], average_precisions[n*i+j+1]), fontsize=16)


# Infer frome Single image
orig_images = []  # Store the images here.
input_images = []  # Store resized versions of the images here.

orig_images.append(imread(single_img_path))
img = image.load_img(single_img_path, target_size=(img_height, img_width))
img = image.img_to_array(img)
input_images.append(img)
input_images = np.array(input_images)

y_pred = model.predict(input_images)

confidence_threshold = 0.5

y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold]
                 for k in range(y_pred.shape[0])]

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_thresh[0])

# Display the image and draw the predicted boxes onto it.
# Set the colors for the bounding boxes
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
plt.figure(figsize=(20, 12))
plt.imshow(orig_images[0])

current_axis = plt.gca()

for box in y_pred_thresh[0]:
    # Transform the predicted bounding boxes for the 512x512 image to the original image dimensions.
    xmin = box[-4] * orig_images[0].shape[1] / img_width
    ymin = box[-3] * orig_images[0].shape[0] / img_height
    xmax = box[-2] * orig_images[0].shape[1] / img_width
    ymax = box[-1] * orig_images[0].shape[0] / img_height
    color = colors[int(box[0])]
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    current_axis.add_patch(plt.Rectangle(
        (xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
    current_axis.text(xmin, ymin, label, size='x-large',
                      color='white', bbox={'facecolor': color, 'alpha': 1.0})


# Create a `BatchGenerator` instance and parse the Pascal VOC labels.
dataset = DataGenerator()

dataset.parse_xml(images_dirs=dataset_images_dir,
                  image_set_filenames=dataset_image_set_filename,
                  annotations_dirs=dataset_annotations_dir,
                  classes=classes,
                  include_classes='all',
                  exclude_truncated=False,
                  exclude_difficult=True,
                  ret=False)

convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

generator = dataset.generate(batch_size=1,
                             shuffle=True,
                             transformations=[convert_to_3_channels,
                                              resize],
                             returns={'processed_images',
                                      'filenames',
                                      'inverse_transform',
                                      'original_images',
                                      'original_labels'},
                             keep_images_without_gt=False)

# Generate a batch and make predictions.

batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(
    generator)

i = 0  # Which batch item to look at

print("Image:", batch_filenames[i])
print()
print("Ground truth boxes:\n")
print(np.array(batch_original_labels[i]))

# Predict.

y_pred = model.predict(batch_images)

confidence_threshold = 0.5

# Perform confidence thresholding.
y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold]
                 for k in range(y_pred.shape[0])]

# Convert the predictions for the original image.
y_pred_thresh_inv = apply_inverse_transforms(
    y_pred_thresh, batch_inverse_transforms)

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_thresh_inv[i])

# Display the image and draw the predicted boxes onto it.

# Set the colors for the bounding boxes
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

plt.figure(figsize=(20, 12))
plt.imshow(batch_original_images[i])

current_axis = plt.gca()

for box in batch_original_labels[i]:
    xmin = box[1]
    ymin = box[2]
    xmax = box[3]
    ymax = box[4]
    label = '{}'.format(classes[int(box[0])])
    current_axis.add_patch(plt.Rectangle(
        (xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))
    current_axis.text(xmin, ymin, label, size='x-large',
                      color='white', bbox={'facecolor': 'green', 'alpha': 1.0})

for box in y_pred_thresh_inv[i]:
    xmin = box[2]
    ymin = box[3]
    xmax = box[4]
    ymax = box[5]
    color = colors[int(box[0])]
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    current_axis.add_patch(plt.Rectangle(
        (xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
    current_axis.text(xmin, ymin, label, size='x-large',
                      color='white', bbox={'facecolor': color, 'alpha': 1.0})
