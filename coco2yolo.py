import json
import os

WEEDCOCO_JSON_PATH = "./dataset/weedcoco.json"
IMAGES_DIRECTORY = "./dataset/images"
OUTPUT_DIRECTORY = "./dataset/annotations"

FILE_EXTENSIONS = ["jpg", "jpeg", "png"]


def truncate(n, decimals=0):
    multiplier = 10**decimals
    return int(n * multiplier) / multiplier


def extract_classes_from_json(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = data.get("categories", [])
    class_names = {category["id"]: category["name"] for category in categories}

    for class_id, class_name in class_names.items():
        print(f"ID: {class_id}, Class: {class_name}")

    return class_names


def coco_to_yolo(bbox, img_width, img_height):
    x_min, y_min, width, height = bbox
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    w = width / img_width
    h = height / img_height

    return truncate(x_center, 6), truncate(y_center, 6), truncate(w, 6), truncate(h, 6)


def convert_weedcoco_to_yolo(json_file, images_dir, output_dir):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    class_names = extract_classes_from_json(json_file)
    images = data.get("images", [])
    annotations = data.get("annotations", [])

    os.makedirs(output_dir, exist_ok=True)

    image_id_to_metadata = {img["id"]: img for img in images}

    for annotation in annotations:
        img_id = annotation["image_id"]
        image_metadata = image_id_to_metadata.get(img_id)

        if image_metadata is None:
            print(f"No metadata found for image ID {img_id}, skipping annotation.")
            continue

        img_filename = image_metadata["file_name"]
        img_width = image_metadata.get("width")
        img_height = image_metadata.get("height")

        if img_width is None or img_height is None:
            print(f"Image dimensions missing for {img_filename}, skipping...")
            continue

        image_path = os.path.join(images_dir, img_filename)
        if not os.path.exists(image_path):
            print(f"Image file {img_filename} not found, skipping...")
            continue

        bbox = annotation["bbox"]
        yolo_bbox = coco_to_yolo(bbox, img_width, img_height)

        category_id = annotation["category_id"]
        yolo_class_id = list(class_names.keys()).index(category_id)

        txt_filename = os.path.splitext(img_filename)[0] + ".txt"
        txt_file_path = os.path.join(output_dir, txt_filename)

        with open(txt_file_path, "a", encoding="utf-8") as f:
            f.write(f"{yolo_class_id} {' '.join(map(str, yolo_bbox))}\n")

    print(f"YOLO annotations saved to {output_dir}")


if __name__ == "__main__":
    convert_weedcoco_to_yolo(WEEDCOCO_JSON_PATH, IMAGES_DIRECTORY, OUTPUT_DIRECTORY)
