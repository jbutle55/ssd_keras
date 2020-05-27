from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd512 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms


def lr_schedule(epoch):
    if epoch < 80:
        return 0.001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001


#  Modified Parameters
n_classes = 80  # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
new_model = True

# If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
initial_epoch = 0
final_epoch = 1
batch_size = 32

# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['background',
           'bus', 'car', 'truck']

# Set all filepaths
weights_path = 'weights/COCO_512.h5'  # If creating new model
model_path = ''  # If loading pre-trained model


train_img_paths = ['/Users/justinbutler/Desktop/test/M0201/JPEGImages']  # The directories that contain the images.
train_annot_paths = ['/Users/justinbutler/Desktop/test/M0201/Annotations']  # The directories that contain the annotations.
train_img_set_paths = ['/Users/justinbutler/Desktop/test/M0201/img_set.txt']  # The paths to the image sets.

valid_img_paths = ['/Users/justinbutler/Desktop/test/M0101/JPEGImages']  # The directories that contain the images.
valid_annot_paths = ['/Users/justinbutler/Desktop/test/M0101/Annotations']  # The directories that contain the annotations.
valid_img_set_paths = ['/Users/justinbutler/Desktop/test/M0101/img_set.txt']  # The paths to the image sets.

#  Fixed Parameters
img_height = 512  # Height of the model input images
img_width = 512  # Width of the model input images
img_channels = 3  # Number of color channels of the model input images
# The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
mean_color = [123, 117, 104]
# The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
swap_channels = [2, 1, 0]
# The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_pascal = [0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05]
# The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales_coco = [0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05]
scales = scales_pascal
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]]  # The anchor box aspect ratios used in the original SSD512; the order matters
two_boxes_for_ar1 = True
# The space between two adjacent anchor box center points for each predictor layer.
steps = [8, 16, 32, 64, 128, 256, 512]
# The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
# Whether or not to clip the anchor boxes to lie entirely within the image boundaries
clip_boxes = False
# The variances by which the encoded target coordinates are divided as in the original implementation
variances = [0.1, 0.1, 0.2, 0.2]
normalize_coords = True

#  Build Model
if new_model is True:
    K.clear_session()  # Clear previous models from memory.

    model = ssd_512(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='training',
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_per_layer=aspect_ratios,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords,
                    subtract_mean=mean_color,
                    swap_channels=swap_channels)

    model.load_weights(weights_path, by_name=True)

    # Instantiate an optimizer and the SSD loss function and compile the model.
    #    If you want to follow the original Caffe implementation, use the preset SGD
    #    optimizer, otherwise I'd recommend the commented-out Adam optimizer.

    #  adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

    model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)

else:
    #  OR Load Model

    # We need to create an SSDLoss object in order to pass that to the model loader.
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

    K.clear_session()  # Clear previous models from memory.

    model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                   'L2Normalization': L2Normalization,
                                                   'compute_loss': ssd_loss.compute_loss})

#  Setup Data Generator
# 1: Instantiate two `DataGenerator` objects: One for training, one for validation.

# Optional: If you have enough memory, consider loading the images into memory for the reasons explained above.

train_dataset = DataGenerator(
    load_images_into_memory=False, hdf5_dataset_path=None)
val_dataset = DataGenerator(
    load_images_into_memory=False, hdf5_dataset_path=None)

# 2: Parse the image and label lists for the training and validation datasets. This can take a while.

train_dataset.parse_xml(images_dirs=train_img_paths,
                        image_set_filenames=train_img_set_paths,
                        annotations_dirs=train_annot_paths,
                        classes=classes,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=False,
                        ret=False)

val_dataset.parse_xml(images_dirs=valid_img_paths,
                      image_set_filenames=valid_img_set_paths,
                      annotations_dirs=valid_annot_paths,
                      classes=classes,
                      include_classes='all',
                      exclude_truncated=False,
                      exclude_difficult=True,
                      ret=False)

# 3: Set the image transformations for pre-processing and data augmentation options.

# For the training generator:
ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                            img_width=img_width,
                                            background=mean_color)

# For the validation generator:
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

# 4: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                   model.get_layer('fc7_mbox_conf').output_shape[1:3],
                   model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv9_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv10_2_mbox_conf').output_shape[1:3]
                   ]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_per_layer=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.5,
                                    normalize_coords=normalize_coords)

# 5: Create the generator handles that will be passed to Keras' `fit_generator()` function.

train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=[
                                             ssd_data_augmentation],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[convert_to_3_channels,
                                                      resize],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size = val_dataset.get_dataset_size()

steps_per_epoch = ceil(train_dataset_size/batch_size)

print("Number of images in the training dataset:\t{:>6}".format(
    train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(
    val_dataset_size))

# Define model callbacks.

# TODO: Set the filepath under which you want to save the model.
model_checkpoint = ModelCheckpoint(filepath='ssd512_pascal_07+12_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)

csv_logger = CSVLogger(filename='ssd300_pascal_07+12_training_log.csv',
                       separator=',',
                       append=True)

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                verbose=1)

terminate_on_nan = TerminateOnNaN()

callbacks = [model_checkpoint,
             csv_logger,
             learning_rate_scheduler,
             terminate_on_nan]

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              callbacks=callbacks,
                              validation_data=val_generator,
                              validation_steps=ceil(
                                  val_dataset_size/batch_size),
                              initial_epoch=initial_epoch)
