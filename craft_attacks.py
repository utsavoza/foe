import argparse
import time
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.stats as st

from copy import copy
from sklearn import metrics
from sklearn.metrics import auc


def mean_ci(x):
    mn = np.mean(x)  # Calculate the mean of the array
    ci = st.t.interval(
        0.95, len(x) - 1, loc=np.mean(x), scale=st.sem(x)
    )  # Calculate the 95% CI
    return mn, ci[0], ci[1]


def deprocess_inception(y, rescale=False):
    x = copy(y).astype(float)
    x += 1.0
    x /= 2.0
    if rescale:
        x *= 255.0
    return x


def preprocess_inception(y):
    x = copy(y).astype(float)
    x /= 255.0
    x *= 2.0
    x -= 1.0
    return x


def print_results(model_preds, y_test):
    acc = np.mean(np.round(model_preds)[:, 0] == y_test[:, 0])
    print("Test accuracy: %0.4f" % acc)

    fpr, tpr, thresholds = metrics.roc_curve(y_test[:, 1], model_preds[:, 1])
    auc_score = auc(fpr, tpr)
    print("AUC: %0.4f" % auc_score)

    conf = mean_ci(np.max(model_preds, axis=1))
    print(
        "Avg. Confidence: "
        + "{0:.6f} ".format(conf[0])
        + "({0:.6f}".format(conf[1])
        + " - {0:.6f})".format(conf[2])
    )


def load_data(
    limit100,
    x_path="data/val_test_x_preprocess.npy",
    y_path="data/val_test_y.npy",
):
    print("Loading Data ....")
    x_test = np.load(x_path, mmap_mode="r")
    y_test = np.load(y_path)

    print("Loaded.")
    if limit100:
        print("Shrinking data to 50 samples per class")
        x_test = np.concatenate((x_test[0:50], x_test[-50:]))
        y_test = np.concatenate((y_test[0:50], y_test[-50:]))

    return x_test, y_test


def build_model(path_model, path_weights=None):
    print("Loading model from disk...")
    model = tf.keras.models.load_model(path_model)

    if path_weights:
        print("Loading model weights...")
        model.load_weights(path_weights)

    print("Loaded model and weights.")
    return model


def create_pgd_attack(
    model,
    x_test,
    y_test,
    eps=0.02,
    eps_iter=0.01,
    nb_iter=20,
    clip_min=-1.0,
    clip_max=1.0,
):
    print("Beginning PGD attack")

    def generate_adversarial_example(x, y, eps):
        """Generate an adversarial example for a single input."""
        x_adv = x
        x_original = x  # Keep a copy of the original input

        for _ in range(nb_iter):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                prediction = model(x_adv)
                loss = tf.keras.losses.categorical_crossentropy(y, prediction)

            # Calculate gradients of loss w.r.t x_adv
            gradients = tape.gradient(loss, x_adv)

            # Update x_adv
            x_adv = x_adv + eps_iter * tf.sign(gradients)

            # Ensure that the perturbation is within the epsilon ball
            x_adv = tf.clip_by_value(x_adv, x_original - eps, x_original + eps)
            x_adv = tf.clip_by_value(x_adv, clip_min, clip_max)

        return x_adv

    t0 = time.time()
    batch_size = 64
    x_test_adv_pgd = np.zeros_like(x_test)

    for i in range(0, x_test.shape[0], batch_size):
        batch_start = i
        batch_end = min(i + batch_size, x_test.shape[0])
        x_batch = tf.convert_to_tensor(x_test[batch_start:batch_end])
        y_batch = tf.convert_to_tensor(y_test[batch_start:batch_end])

        if not (i % (20 * batch_size)):
            print(f"Attacking batch from {batch_start} to {batch_end}", file=sys.stderr)

        x_batch_adv = generate_adversarial_example(x_batch, y_batch, eps)
        x_test_adv_pgd[batch_start:batch_end] = x_batch_adv.numpy()

    t1 = time.time()
    total = t1 - t0
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    print(f"Completed attack in {h:.0f}:{m:02.0f}:{s:02.0f}")

    return x_test_adv_pgd


def save_image(file_path, image):
    plt.imsave(file_path, image)


def evaluate(
    model,
    x_test,
    y_test,
    x_test_adv,
    attack_type,
    print_image_index=[],
    test_clean=False,
    save_flag=True,
):
    # Optionally test on clean examples
    if test_clean:
        print("Clean Examples:")
        model_preds_clean = model.predict(x_test, batch_size=32)
        print_results(model_preds_clean, y_test)
        print("")

    # Evaluate results on adversarial examples
    print("Adversarial Examples:")
    model_preds = model.predict(x_test_adv, batch_size=32)
    print_results(model_preds, y_test)

    if save_flag:
        np.save("data/pgd_preds_transfer_" + attack_type + ".npy", model_preds)

    # Calculate L2 norm of perturbations
    l2_norm = np.sqrt(np.sum((x_test_adv - x_test) ** 2, axis=(1, 2, 3)))
    l2_norm_sum = mean_ci(l2_norm)
    print(
        f"Avg. L2 norm of perturbations: {l2_norm_sum[0]:.6f} ({l2_norm_sum[1]:.6f} - {l2_norm_sum[2]:.6f})"
    )

    # Identify the most perturbed images from healthy and sick patients
    ind_max_diff_healthy = np.argmax(l2_norm[y_test[:, 1] == 0])
    ind_max_diff_sick = np.argmax(l2_norm[y_test[:, 1] == 1])
    ind_max_diff_sick_shifted = ind_max_diff_sick + np.nonzero(y_test[:, 1] == 1)[0][0]
    print(
        f"Most perturbed images are {ind_max_diff_healthy} and {ind_max_diff_sick_shifted}"
    )

    # Optionally save the most perturbed images and any images whose indices are in print_image_index
    if save_flag:
        for ind in print_image_index:
            save_image(
                f"images/normal_transfer_img_{ind}.png",
                deprocess_inception(x_test[ind]),
            )
            save_image(
                f"images/attack_transfer_pgd_img{ind}{attack_type}.png",
                deprocess_inception(x_test_adv[ind]),
            )

        save_image(
            f"images/biggest_transfer_attack_{attack_type}_img{ind_max_diff_healthy}.png",
            deprocess_inception(x_test_adv[ind_max_diff_healthy]),
        )
        save_image(
            f"images/biggest_transfer_attack_{attack_type}_img{ind_max_diff_sick_shifted}.png",
            deprocess_inception(x_test_adv[ind_max_diff_sick_shifted]),
        )


def run():
    parser = argparse.ArgumentParser(description="Build adversarial attacks")
    parser.add_argument(
        "--path_model",
        type=str,
        default="models/ResNet50_e10_LearnRate-0.001_weights.epoch-01-val_acc-0.5000.hdf5",
        help="Path to .h5 file for true model (default: models/wb_model.h5)",
    )
    parser.add_argument(
        "--path_wb_weights",
        type=str,
        default=None,  # models/wb_weights.hdf5
        help="Optionally provide path to weights to load into WB model (default: None)",
    )
    parser.add_argument(
        "--path_bb_weights",
        type=str,
        default=None,
        help="Path to weights for independent BB model (default: models/bb_weights.hdf5)",
    )
    parser.add_argument(
        "--path_data_x",
        type=str,
        default="data/val_transfer_x.npy",
        help="Path to training data X (default: data/val_test_x_preprocess.npy)",
    )
    parser.add_argument(
        "--path_data_y",
        type=str,
        default="data/val_transfer_y.npy",
        help="Path to training labels y (default: data/val_test_y.npy)",
    )
    parser.add_argument(
        "--limit100",
        action="store_true",
        default=False,
        help="Only attack 50 examples from each class",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.02,
        help="Epsilon parameter of PGD (default: 0.02)",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="data/pgd_eps02_transfer",
        help="Base of filename to save model (default: 'data/pgd_eps02_')",
    )
    parser.add_argument(
        "--print_image_index",
        type=int,
        nargs="*",
        default=None,
        help="Image indices to save to file in addition to the most perturbed. (default: None)",
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        default=True,
        help="Save results (default: true)",
    )

    # Get Arguments
    args = parser.parse_args()
    eps = args.eps
    filename = args.filename
    path_model = args.path_model
    path_bb_weights = args.path_bb_weights
    path_wb_weights = args.path_wb_weights
    path_data_x = args.path_data_x
    path_data_y = args.path_data_y
    limit100 = args.limit100
    save_results = args.save_results
    print_image_index = args.print_image_index
    if limit100:
        filename += "small_transfer_"

    # Run attack
    (x_test, y_test) = load_data(limit100, x_path=path_data_x, y_path=path_data_y)

    # Black box
    print("\nBlack Box Attack:")
    model = build_model(path_model, path_bb_weights)
    x_test_adv_bb = create_pgd_attack(model, x_test, y_test, eps)
    if save_results:
        print("Saving to : ", filename + "white_box_transfer.npy")
        np.save(filename + "black_box_transfer.npy", x_test_adv_bb)

    # White box
    print("\n\nWhite Box Attack:")
    model = build_model(path_model, path_wb_weights)
    x_test_adv_wb = create_pgd_attack(model, x_test, y_test, eps)
    if save_results:
        print("Saving to : ", filename + "white_box_transfer.npy")
        np.save(filename + "white_box_transfer.npy", x_test_adv_wb)

    # Evaluate
    print("\n_____Evaluate White Box Attack_____")
    evaluate(
        model,
        x_test,
        y_test,
        x_test_adv_wb,
        "pgd_white_box",
        test_clean=True,
        save_flag=save_results,
    )  # print_image_index

    print("\n_____Evaluate Black Box Attack_____")
    evaluate(
        model,
        x_test,
        y_test,
        x_test_adv_bb,
        "pgd_black_box",
        save_flag=save_results,
    )  # print_image_index


if __name__ == "__main__":
    run()
