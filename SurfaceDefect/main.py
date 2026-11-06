from anomalib.deploy import OpenVINOInferencer
import time
import matplotlib.pyplot as plt
import os

inferencer = OpenVINOInferencer(
    path="./model.bin",  # Path to the OpenVINO IR model.
    metadata="./metadata.json",  # Path to the metadata file.
    device="CPU",  # We would like to run it on an Intel CPU.
)

# image_path = "./test/test_3.png"
defaultSavePath = "D:\\result\\"


def predict_image(image_path, save_path=defaultSavePath):
    t = time.time()
    predictions = inferencer.predict(image=image_path)
    print("Thời gian chạy: ", time.time() - t)

    print(predictions.pred_score, predictions.pred_label)
    plt.subplot(2, 2, 1)
    plt.title("Input image")
    plt.axis("off")
    plt.imshow(predictions.image)

    plt.subplot(2, 2, 2)
    plt.title("Anomaly map")
    plt.axis("off")
    plt.imshow(predictions.anomaly_map)

    plt.subplot(2, 2, 3)
    plt.title("Heat map")
    plt.axis("off")
    plt.imshow(predictions.heat_map)

    plt.subplot(2, 2, 4)
    plt.title("Prediction")
    plt.axis("off")
    plt.imshow(predictions.segmentations)

    plt.savefig(save_path + os.path.basename(image_path))

    # plt.show()


def checkPath(path):
    png_filenames = []

    if os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.endswith(".png"):
                png_filenames.append(os.path.join(path, filename))

        if len(png_filenames) == 0:
            print("No image.png in folder")
            return False, png_filenames
        else:
            return True, png_filenames

    elif os.path.isfile(path):
        if path.endswith(".png"):
            png_filenames.append(path)
            return True, png_filenames
        else:
            print("path is not a image.png")
            return False, png_filenames

    else:
        print("path is not a image.png or a folder containing image.png")
        return False, png_filenames


stop = False

while not stop:
    path = input("Enter a path of a image.png or a folder: ")
    c, image_names = checkPath(path)
    while not c:
        path = input("Enter a path of a image.png or a folder: ")
        c, image_names = checkPath(path)
    for image_name in image_names:
        predict_image(image_name)
    plt.show()

    stop = float(input("Do you want to stop? "))
