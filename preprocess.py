import os
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import copy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def preprocess_image(img_arr):
    x = copy(img_arr).astype(float)
    x /= 255.0
    x *= 2.0
    x -= 1.0
    return x


def process_images(image_paths, target_size=(224, 224)):
    images = []
    for img_path in image_paths:
        try:
            img = load_img(img_path, target_size=target_size)
            img_array = img_to_array(img)
            img_array = preprocess_image(img_array)
            images.append(img_array)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
    return images


def log_progress(futures, total, desc="Processing Images", interval=5):
    """
    Log progress at regular intervals.

    :param futures: A list of Future objects.
    :param total: Total number of tasks.
    :param desc: Description of the task.
    :param interval: Time interval (in seconds) between log messages.
    """
    start_time = time.time()
    completed = 0

    while completed < total:
        # Wait for the interval period
        time.sleep(interval)

        # Update the number of completed tasks
        completed = sum(f.done() for f in futures)

        # Calculate elapsed time and estimated time remaining
        elapsed_time = time.time() - start_time
        remaining_time = (
            (elapsed_time / completed) * (total - completed) if completed else 0
        )

        # Log the progress
        print(
            f"{desc}: {completed}/{total} tasks completed. "
            f"Elapsed time: {elapsed_time:.2f}s, "
            f"Estimated time remaining: {remaining_time:.2f}s."
        )


def batch_process(data_dir, num_workers=4):
    image_paths = []
    labels = []

    for category in ["Cardiomegaly", "No_Finding"]:
        category_dir = os.path.join(data_dir, category)
        label = 1 if category == "Cardiomegaly" else 0

        for img_name in os.listdir(category_dir):
            img_path = os.path.join(category_dir, img_name)
            image_paths.append(img_path)
            labels.append(label)

    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=2)

    # Splitting paths into chunks for parallel processing
    chunk_size = len(image_paths) // num_workers
    image_path_chunks = [
        image_paths[i : i + chunk_size] for i in range(0, len(image_paths), chunk_size)
    ]
    if len(image_path_chunks) > num_workers:
        # If there are more chunks than workers, merge the last two chunks
        image_path_chunks[-2].extend(image_path_chunks[-1])
        image_path_chunks.pop()

    images = []
    futures = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all the tasks and record the future objects
        for chunk in image_path_chunks:
            future = executor.submit(process_images, chunk)
            futures.append(future)

        # Log progress instead of using tqdm
        log_progress(futures, len(futures), desc="Processing Images")

        # Create a progress bar and iterate over the future objects as they complete
        for future in as_completed(futures):
            chunk_images = future.result()
            images.extend(chunk_images)

    return np.array(images), labels


if __name__ == "__main__":
    # train_images, train_labels = batch_process("data/images/train", num_workers=16)
    val_images, val_labels = batch_process("data/transfer/images/val", num_workers=4)

    # np.save("data/train_x.npy", train_images)
    # np.save("data/train_y.npy", train_labels)
    np.save("data/val_transfer_x.npy", val_images)
    np.save("data/val_transfer_y.npy", val_labels)
