from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
import os
import argparse
from datetime import datetime as dt

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

from eval_utils.average_precision_evaluator import Evaluator



def lr_schedule(epoch):
    if epoch < 80:
        return 0.001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001


def main(args):
    #  Modified Parameters
    # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
    n_classes = args.num_classes
    skip_training = args.no_train

    # If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
    initial_epoch = args.initial_epoch
    final_epoch = args.initial_epoch + args.epochs
    batch_size = args.batch_size

    # The XML parser needs to now what object class names to look for and in which order to map them to integers.
    classes = ['background']
    for cl in args.classes:
        classes.append(cl)
    print('Classes used: {}'.format(classes))

    # Set all filepaths
    print(os.getcwd())
    weights_path = args.weights_path  # If creating new model
    model_path = args.model_path  # If loading pre-trained model

    # The directories that contain the images.
    train_img_paths = args.train_jpeg
    # The directories that contain the annotations.
    train_annot_paths = args.train_annot
    train_img_set_paths = args.train_set  # The paths to the image sets.

    # The directories that contain the images.
    valid_img_paths = args.valid_jpeg
    # The directories that contain the annotations.
    valid_annot_paths = args.valid_annot
    valid_img_set_paths = args.valid_set  # The paths to the image sets.

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
    if skip_training is False:
        if args.new_model:
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

            # initial_epoch = int(model_path[-6:-3]) + 1

            # We need to create an SSDLoss object in order to pass that to the model loader.
            ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

            K.clear_session()  # Clear previous models from memory.

            model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                           'L2Normalization': L2Normalization,
                                                           'compute_loss': ssd_loss.compute_loss})

        # 1: Instantiate two `DataGenerator` objects: One for training, one for validation.
        #  Setup Data Generator
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
                           model.get_layer(
                               'conv6_2_mbox_conf').output_shape[1:3],
                           model.get_layer(
                               'conv7_2_mbox_conf').output_shape[1:3],
                           model.get_layer(
                               'conv8_2_mbox_conf').output_shape[1:3],
                           model.get_layer(
                               'conv9_2_mbox_conf').output_shape[1:3],
                           model.get_layer(
                               'conv10_2_mbox_conf').output_shape[1:3]
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

        # callbacks = [model_checkpoint,
        #             csv_logger,
        #             learning_rate_scheduler,
        #             terminate_on_nan]

        callbacks = [learning_rate_scheduler,
                     terminate_on_nan]

        history = model.fit_generator(generator=train_generator,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=final_epoch,
                                      callbacks=callbacks,
                                      validation_data=val_generator,
                                      validation_steps=ceil(
                                          val_dataset_size/batch_size),
                                      initial_epoch=initial_epoch)

        print('Training Complete')

        model.save(args.saved_model)
        print('Saving model to: {}'.format(args.saved_model))

    # Evaluate model
    if args.validate:
        # Set a few configuration parameters.
        model_mode = 'training'  # Or training

        # For image for inference
        single_img_path = '/kaggle/input/aerial/aerial_yolo/aerial_yolo/valid/JPEGImages/DJI_0019.jpg'

        # Load model
        # TODO: Set the path to the `.h5` file of the model to be loaded.
        model_path = args.saved_model

        # We need to create an SSDLoss object in order to pass that to the model loader.
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

        K.clear_session()  # Clear previous models from memory.

        model_objects = {'AnchorBoxes': AnchorBoxes,
                         'L2Normalization': L2Normalization,
                         'DecodeDetections': DecodeDetections,
                         'compute_loss': ssd_loss.compute_loss}

        model = load_model(model_path, custom_objects=model_objects)

        dataset = DataGenerator()

        dataset.parse_xml(images_dirs=valid_img_paths,
                          image_set_filenames=valid_img_set_paths,
                          annotations_dirs=valid_annot_paths,
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

        # Format of predictions: Nested list of each class
        # pred_list[0] - Background, pred_list[1] - List of class 1 predictions, ...
        # Each prediction format: [Image id, confidence, xmin, ymin, xmax, ymax]
        pred_list = evaluator.prediction_results

        # Determine number of predictions for each class above a set threshold (probably 0.5)
        filt_1 = [i for i in pred_list[1] if float(i[1]) >= 0.5]
        filt_2 = [i for i in pred_list[2] if float(i[1]) >= 0.5]
        filt_3 = [i for i in pred_list[3] if float(i[1]) >= 0.5]

        print('Class 1: Total predictions - {}, Theshold predictions - {}'.format(
            len(pred_list[1]), len(filt_1)))
        print('Class 2: Total predictions - {}, Theshold predictions - {}'.format(
            len(pred_list[2]), len(filt_2)))
        print('Class 3: Total predictions - {}, Theshold predictions - {}'.format(
            len(pred_list[3]), len(filt_3)))

        # Display results
        for i in range(1, len(average_precisions)):
            print("{:<14}{:<6}{}".format(
                classes[i], 'AP', round(average_precisions[i], 3)))
        print()
        print("{:<14}{:<6}{}".format(
            '', 'mAP', round(mean_average_precision, 3)))

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

        # Create a `BatchGenerator` instance and parse the Pascal VOC labels.
        dataset = DataGenerator()

        dataset.parse_xml(images_dirs=valid_img_paths,
                          image_set_filenames=valid_img_set_paths,
                          annotations_dirs=valid_annot_paths,
                          classes=classes,
                          include_classes='all',
                          exclude_truncated=False,
                          exclude_difficult=False,
                          ret=False)

        convert_to_3_channels = ConvertTo3Channels()
        resize = Resize(height=img_height, width=img_width)

        generator = dataset.generate(batch_size=50,
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

        for i in range(0, len(batch_filenames)):

            print("Image:", batch_filenames[i])
            print()
            print("Ground truth boxes:")
            print(len(np.array(batch_original_labels[i])))

            batch_file_id = batch_filenames[i][-13:-4]

            # Filter list of predictions for image we are looking at
            filtered_preds = [np.hstack(np.asarray([x, y[1:]])).tolist() for x in range(
                len(pred_list)) for y in pred_list[x] if y[0] == batch_file_id]

            print('Total number of predictions: {}'.format(len(filtered_preds)))

            if len(filtered_preds) == 0:
                'Skipping - No predictions made for image'
                continue

            confidence_threshold = 0.5
            thresh_preds = [x for x in filtered_preds if x[1]
                            > confidence_threshold]

            print('Total number of predictions with threshold larger than 0.5: {}'.format(
                thresh_preds))

            np.set_printoptions(precision=2, suppress=True, linewidth=75)
            print("\nPredicted boxes:")
            print('   class   conf xmin   ymin   xmax   ymax')
            print(np.array(thresh_preds))

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

            for box in thresh_preds:
                xmin = box[2]
                ymin = box[3]
                xmax = box[4]
                ymax = box[5]
                color = colors[int(box[0])]
                print('color')
                label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
                current_axis.add_patch(plt.Rectangle(
                    (xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
                current_axis.text(xmin, ymin, label, size='x-large',
                                  color='yellow', bbox={'facecolor': color, 'alpha': 1.0})

            plt.savefig('output_{}.png'.format(i))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train_jpeg', '-tj', nargs='*', help='')
    parser.add_argument('--train_annot', '-ta', nargs='*', help='')
    parser.add_argument('--train_set', '-ts', nargs='*', help='')
    parser.add_argument('--valid_jpeg', '-vj', nargs='*', help='')
    parser.add_argument('--valid_annot', '-va', nargs='*', help='')
    parser.add_argument('--valid_set', '-vs', nargs='*', help='')
    parser.add_argument('--initial_epoch', '-ie', default=0, help='')
    parser.add_argument('--weights_path', help='')
    parser.add_argument('--model_path', '-mp', help='')
    parser.add_argument('--saved_model', '-save',
                        default='trained_ss.h5', help='')
    parser.add_argument('--classes', nargs='*', help='')
    parser.add_argument('--num_classes', type=int, help='')
    parser.add_argument('--epochs', type=int, default=100, help='')
    parser.add_argument('--mode', default='fit', help='')
    parser.add_argument('--transfer', default='darknet',
                        required=False, help='')
    parser.add_argument('--batch_size', type=int,
                        default=16, required=False, help='')
    parser.add_argument('--num_weight_class', type=int, default=80,
                        required=False,
                        help='The number of classes in the backbone weights. 80 for COCO.')
    parser.add_argument('--no_train', '-nt', action='store_true', help='')
    parser.add_argument('--validate', '-v', action='store_true', help='')
    parser.add_argument('--new_model', '-nm', action='store_true', help='')

    args = parser.parse_args()
    main(args)
