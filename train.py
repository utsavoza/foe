import os
import argparse
import numpy as np

from mixup_generator import MixupGenerator

import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.applications import InceptionV3, ResNet50
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input


def load_data(batch_size, mixup, v_flip, rotation):
    train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=rotation,
        width_shift_range=0.2,
        height_shift_range=0.2,
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        vertical_flip=v_flip,
    )

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Mixup
    if mixup:
        x_train = np.load("data/train_x.npy")  # N x 224 x 224 x 3
        y_train = np.load("data/train_y.npy")
        x_test = np.load("data/val_x.npy")  # N x 224 x 224 x 3
        y_test = np.load("data/val_y.npy")

        train_generator = MixupGenerator(
            x_train, y_train, batch_size=batch_size, alpha=0.2, datagen=train_datagen
        )()
        validation_generator = test_datagen.flow(x_test, y_test, batch_size=batch_size)
        n_data = (x_train.shape[0], x_test.shape[0])
    else:
        train_generator = train_datagen.flow_from_directory(
            "data/images/train",
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
        )
        validation_generator = test_datagen.flow_from_directory(
            "data/images/val",
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
        )
        n_data = (train_generator.n, validation_generator.n)

    return train_generator, validation_generator, n_data


def construct_model(inception_model, learning_rate, freeze_early_layers=False):
    # Base Model
    if inception_model:
        base_model = InceptionV3(weights="imagenet", include_top=True)
    else:
        base_model = ResNet50(weights="imagenet", include_top=True)

    outputs = base_model.layers[-2].output

    # Finetune Layer
    fine_tune_layer = Dense(128)(outputs)
    fine_tune_layer = Dropout(0.2)(fine_tune_layer)  # usually 0.2
    fine_tune_layer = Dense(2, activation="softmax")(fine_tune_layer)

    # Final Model
    model = Model(inputs=base_model.input, outputs=fine_tune_layer)

    # Freeze early layers
    if freeze_early_layers:
        for layer in model.layers[:25]:
            layer.trainable = False

    # Compile Model
    model.compile(
        optimizer=optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def generate_callbacks(
    inception_model,
    name_append,
    learning_rate,
    model_checking,
    model_checkpoint_period,
    early_stopping,
    early_stop_patience,
):
    # Construct TB Directory
    if inception_model:
        run_id = "InceptionV3"
    else:
        run_id = "ResNet50"
    run_id += "_" + name_append
    run_id += "_LearnRate-" + str(learning_rate)

    tb_dir0 = "./keras_logs/" + run_id + "_upfront"
    if not os.path.exists(tb_dir0):
        os.makedirs(tb_dir0)
    tb_dir1 = "./keras_logs/" + run_id + "_wCheckOrStop"
    if not os.path.exists(tb_dir1):
        os.makedirs(tb_dir1)

    # Tensorboard Callback
    tb_callback_0 = callbacks.TensorBoard(
        log_dir=tb_dir0, histogram_freq=0, write_graph=False, write_images=False
    )
    tb_callback_1 = callbacks.TensorBoard(
        log_dir=tb_dir1, histogram_freq=0, write_graph=False, write_images=False
    )
    callback_list = [tb_callback_1]

    # Model Checking Callback
    if model_checking:
        model_check = callbacks.ModelCheckpoint(
            "models/"
            + run_id
            + "_weights.epoch-{epoch:02d}-val_acc-{val_accuracy:.4f}.hdf5",
            monitor="val_accuracy",
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            save_freq="epoch",
        )
        callback_list += [model_check]

    # Early Stopping Callback
    if early_stopping:
        early_stop = callbacks.EarlyStopping(
            monitor="val_accuracy",
            min_delta=0,
            patience=early_stop_patience,
            verbose=0,
            mode="auto",
        )

        callback_list += [early_stop]

    return [tb_callback_0], callback_list


def fit_model(
    model,
    callbacks,
    train_generator,
    validation_generator,
    n_data,
    batch_size,
    max_epochs,
    n_epoch_before_saving,
):
    model.fit(
        train_generator,
        steps_per_epoch=n_data[0] // batch_size,
        epochs=n_epoch_before_saving,
        validation_data=validation_generator,
        validation_steps=n_data[1] // batch_size,
        callbacks=callbacks[0],  # Begin with only tensorboard
    )

    if max_epochs - n_epoch_before_saving > 0:
        model.fit(
            train_generator,
            steps_per_epoch=n_data[0] // batch_size,
            epochs=max_epochs - n_epoch_before_saving,
            validation_data=validation_generator,
            validation_steps=n_data[1] // batch_size,
            callbacks=callbacks[1],
        )

    return model


def run():
    # Setup argparse
    parser = argparse.ArgumentParser(
        description="Train ResNet50 or InceptionResNetV2 Model"
    )
    parser.add_argument(
        "--incept",
        action="store_true",
        default=False,
        help="Train Inception Model, else train ResNet model (default: False)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=300,
        help="Max number epochs for which to train. (default: 300)",
    )
    parser.add_argument(
        "--model_checkpointing",
        type=int,
        default=10,
        help="Save best model checkpoints with period N.  Negative value = no checkpointing. (default: 10)",
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=100,
        help="Apply early stopping with patience N. Negative value = no early stopping (default: 100)",
    )
    parser.add_argument(
        "--min_epochs",
        type=int,
        default=200,
        help="Number of epochs to run before checkpoint or early stopping (default: 200)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size  (default: 32)"
    )
    parser.add_argument("--name_append", type=str, default="", help="Add-on to name")
    parser.add_argument(
        "--gpu1",
        action="store_true",
        default=False,
        help="Use GPU 1, else use GPU zero",
    )
    parser.add_argument(
        "--mixup",
        action="store_true",
        default=False,
        help="Use mixup for data processing",
    )
    parser.add_argument(
        "--v_flip",
        action="store_true",
        default=False,
        help="Vertical Flipping during data aug",
    )
    parser.add_argument(
        "--rotation", type=int, default=45, help="Degree of rotation during data aug"
    )

    # Parse arguments
    args = parser.parse_args()
    inception_model = args.incept
    learning_rate = args.lr
    max_epochs = args.max_epochs
    batch_size = args.batch_size
    model_checkpoint_period = args.model_checkpointing
    early_stop_patience = args.early_stopping
    n_epoch_before_saving = args.min_epochs
    mixup = args.mixup
    name_append = args.name_append
    v_flip = args.v_flip
    rotation = args.rotation

    # Set CUDA Device Using Flag
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.gpu1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    devices = tf.config.list_physical_devices()
    print("Available devices:", devices)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

    # Handle checkpointing and early stopping
    model_checkpointing = model_checkpoint_period >= 0
    if model_checkpointing:
        print("Checkpointing models with period ", model_checkpoint_period)

    early_stopping = early_stop_patience >= 0
    if early_stopping:
        print("Applying early stopping with patience ", early_stop_patience)

    if not (model_checkpointing or early_stopping):
        n_epoch_before_saving = 0
    else:
        print(
            "Waiting",
            n_epoch_before_saving,
            "epochs before starting checkpointing and/or saving.",
        )

    # Get Model Type
    if inception_model:
        print("Using InceptionV3architecture")
        model_name = "models/InceptionV3"
    else:
        print("Using ResNet50 architecture")
        model_name = "models/ResNet50"

    if name_append != "":
        model_name += "_" + name_append
    model_name += (
        "_LR-"
        + str(learning_rate)
        + "_max_epochs-"
        + str(max_epochs)
        + "_weights_final.hdf5"
    )

    print("filename for save weights: ", model_name)
    print("\n")

    # Train the model
    train_generator, validation_generator, n_data = load_data(
        batch_size, mixup, v_flip, rotation
    )
    model = construct_model(inception_model, learning_rate)
    v_callbacks = generate_callbacks(
        inception_model,
        name_append,
        learning_rate,
        model_checkpointing,
        model_checkpoint_period,
        early_stopping,
        early_stop_patience,
    )
    model = fit_model(
        model,
        v_callbacks,
        train_generator,
        validation_generator,
        n_data,
        batch_size,
        max_epochs,
        n_epoch_before_saving,
    )
    model.save_weights(model_name)


if __name__ == "__main__":
    run()
