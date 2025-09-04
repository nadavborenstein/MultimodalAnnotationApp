from flask import Flask, render_template, request, jsonify
import os, random
import pandas as pd
from glob import glob
from typing import List, Dict

app = Flask(__name__)

# Paths
IMAGE_FOLDER = "static/images/"
NOTES = "/Users/knf792/gits/AnnotationWebsite/data/tweets_with_images.csv"
ANNOTATIONS_FILE = "data/annotations.csv"

# Load texts


def select_batch(data: pd.DataFrame) -> pd.DataFrame:
    images = glob(os.path.join(IMAGE_FOLDER, "*.png"))
    print(f"Total images found: {len(images)}")
    image_names = [os.path.basename(img) for img in images]
    print(f"Sample image names: {image_names[:5]}")
    data = data[data["image_name"].isin(image_names)]
    
    print(f"Number of available images: {len(data)}")
    sampled_data = data.sample(n=5, random_state=42)
    return sampled_data


def extract_data_from_note(note: pd.Series) -> Dict:
    text = note["full_text"]
    image_name = note["image_name"]
    image_path = os.path.join(IMAGE_FOLDER, image_name)

    id = note["tweetId"]
    return {
        "ID": id,
        "Tweet": text,
        "ImageURL": image_path,
    }


def select_row(data: pd.DataFrame) -> pd.Series:
    index = random.randint(0, len(data) - 1)
    note = data.iloc[index]
    return note


data = pd.read_csv(NOTES)
data = select_batch(data)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_task")
def get_task():
    # Pick random image and text
    note = select_row(data)
    note_content = extract_data_from_note(note)
    return jsonify({"image": note_content["ImageURL"], "text": note_content["Tweet"]})


@app.route("/submit", methods=["POST"])
def submit():
    data = request.json
    data = pd.DataFrame([data])

    data.to_csv(
        ANNOTATIONS_FILE,
        mode="a",
        header=not os.path.exists(ANNOTATIONS_FILE),
        index=False,
    )

    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
