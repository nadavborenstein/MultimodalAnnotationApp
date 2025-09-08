import streamlit as st
import random
import os
import pandas as pd
from glob import glob
from st_files_connection import FilesConnection
import io
from PIL import Image
from collections import Counter
from time import time

st.set_page_config(layout="wide")
conn = st.connection("gcs", type=FilesConnection)


NOTES = "annotation-experiment/data/tweets_with_images.csv"
MAX_ANNOTATIONS_PER_WORKER = 20  # TODO: adjust as needed
ID_COL = "tweetId"
IMAGE_FOLDER = "annotation-experiment/static/resized_images/"
PROGRESS_FOLDER = "annotation-experiment/data/worker_progress"
DONE_FILE = "annotation-experiment/data/done.txt"
NON_PARTICIPANTS_FILE = "annotation-experiment/data/non_participants.txt"
NUM_ANNOTATORS_PER_ITEM = 3  # TODO: adjust as needed
POSITIVE_EMOTIONS = ["hope", "joy", "pride", "curiosity"]
NEGATIVE_EMOTIONS = ["fear", "anger", "sadness", "ridicule"]


def time_before():
    return int(time() * 1000)


def timeit(start_time):
    return int(time() * 1000) - start_time


def append_to_file(item: str, file_path: str):
    done = conn.fs.open(file_path, "r").read()
    done += f"{item}\n"
    conn.fs.open(file_path, "w").write(done)


def record_non_participation():
    if not st.session_state.worker_id:
        return
    if conn.fs.exists(NON_PARTICIPANTS_FILE):
        append_to_file(st.session_state.worker_id, NON_PARTICIPANTS_FILE)
    else:
        conn.fs.open(NON_PARTICIPANTS_FILE, "w").write(
            f"{st.session_state.worker_id}\n"
        )
    st.success("Your choice has been recorded. Thank you.")


@st.cache_resource
def load_notes() -> pd.DataFrame:
    notes = conn.fs.open(NOTES, "r").read()
    notes = pd.read_csv(io.StringIO(notes))
    # seed from worker_id
    images = conn.fs.glob(f"{IMAGE_FOLDER}*.png")
    image_names = [os.path.basename(img) for img in images]
    notes = notes[notes["image_name"].isin(image_names)]
    notes = notes.drop_duplicates(subset=["image_name"])
    notes = notes.head(20)
    notes.set_index(ID_COL, inplace=True, drop=False)
    return notes


def load_done() -> set:
    done = conn.fs.open(DONE_FILE, "r").read()
    done = done.split("\n")
    done = [d.strip() for d in done if d]
    counts = Counter(done)
    done = {d for d, c in counts.items() if c >= NUM_ANNOTATORS_PER_ITEM}

    return done


@st.cache_data
def loading_images(image_names) -> list:
    images = dict()
    for image_name in image_names:
        image_path = os.path.join(IMAGE_FOLDER, image_name)
        image_data = conn.fs.open(image_path, "rb").read()
        images[image_name] = image_data
    return images


@st.cache_resource
def get_worker_session(worker_id: str, notes: pd.DataFrame) -> pd.DataFrame:
    # check if a progress file exists for this worker
    progress_file = f"{PROGRESS_FOLDER}/progress_{worker_id}.csv"
    if conn.fs.exists(progress_file):
        progress = conn.fs.open(progress_file, "r").read()
        progress = pd.read_csv(io.StringIO(progress))
        progress.set_index(ID_COL, inplace=True, drop=False)
        return progress
    else:
        seed = hash(st.session_state.worker_id) % (2**31)
        done_notes = load_done()
        notes = notes[~notes.index.isin(done_notes)]

        notes_to_label = notes.sample(
            n=min(MAX_ANNOTATIONS_PER_WORKER, len(notes)), random_state=seed
        )
        ids_to_label = notes_to_label.index.tolist()
        progress = pd.DataFrame(
            {
                ID_COL: ids_to_label,
                "worker_id": [worker_id] * len(ids_to_label),
                "done": [None] * len(ids_to_label),
                "label": [None] * len(ids_to_label),
                "image_name": notes_to_label["image_name"].tolist(),
            }
        )
        progress.set_index(ID_COL, inplace=True, drop=False)
        s = progress.to_csv(index=False)
        conn.fs.open(progress_file, "w").write(s)
        return progress


def get_item_number(progress: pd.DataFrame) -> int:
    done = progress["done"].notnull().sum()
    return done + 1


def select_next_item_for_worker_id(progress: pd.DataFrame) -> str:
    # select the next item that is not done
    not_done = progress[progress["done"].isnull()]
    if not_done.empty:
        return None
    next_id = not_done.index[0]
    return next_id


def clear_selections():
    """
    Clear all selections in the session state.
    """
    for key in POSITIVE_EMOTIONS + NEGATIVE_EMOTIONS + ["none"]:
        if key in st.session_state:
            st.session_state[key] = False
    for key in ["other_positive", "other_negative"]:
        if key in st.session_state:
            st.session_state[key] = ""

    if "emotion_label" in st.session_state:
        st.session_state["emotion_label"] = []


def collect_selected_labels() -> list:
    """
    Collect selected labels from the session state.
    Returns a list of selected labels.
    """
    selected_labels = []
    for emotion in POSITIVE_EMOTIONS + NEGATIVE_EMOTIONS + ["none"]:
        if st.session_state.get(emotion, False):
            selected_labels.append(emotion)
    other_positive = st.session_state.get("other_positive", "")
    other_negative = st.session_state.get("other_negative", "")
    if other_positive:
        other_positive_labels = [
            label.strip()
            for label in other_positive.split(",")
            if label.strip() and label.strip().lower() not in selected_labels
        ]
        selected_labels.extend(other_positive_labels)
    if other_negative:
        other_negative_labels = [
            label.strip()
            for label in other_negative.split(",")
            if label.strip() and label.strip().lower() not in selected_labels
        ]
        selected_labels.extend(other_negative_labels)

    return selected_labels


def confirm_label(note: pd.Series):
    """
    Confirm the selected label and update the progress.
    """
    progress_file = f"{PROGRESS_FOLDER}/progress_{st.session_state.worker_id}.csv"
    selected_labels = collect_selected_labels()

    if not selected_labels:
        return

    index = note[ID_COL]
    st.session_state.progress.at[index, "done"] = True
    st.session_state.progress.at[index, "label"] = str(selected_labels)
    clear_selections()
    s = st.session_state.progress.to_csv(index=False)
    conn.fs.open(progress_file, "w").write(s)
    append_to_file(index, DONE_FILE)


st.title("Annotation experiment")

if "worker_id" not in st.session_state:
    placeholder = "ID"
else:
    placeholder = st.session_state.worker_id

st.text_input(
    "Please enter your Prolific ID",
    key="worker_id",
    placeholder=placeholder,
    value=st.session_state.get("worker_id", ""),
    disabled="worker_id" in st.session_state and len(st.session_state.worker_id),
    help="Your Prolific ID is used to track your progress and ensure you do not annotate the same item multiple times.",
)
if not st.session_state.worker_id:
    st.warning("Please enter your Prolific ID to proceed.")
    st.stop()
else:
    st.success(
        f"Thank you for providing your Prolific ID: {st.session_state.worker_id}."
    )

st.header("Consent")
st.radio(
    label="Do you consent to participate in this study?",
    options=["", "No", "Yes"],
    key="consent",
    horizontal=True,
    index=0,
    help="You must consent to participate in this study to proceed.",
    on_change=lambda: st.session_state.update({"show_consent": False}),
    disabled="consent" in st.session_state
    and st.session_state.consent in ["Yes", "No"],
)

if st.session_state.consent == "Yes":
    st.session_state.show_consent = False
    st.success(
        "Thank you for consenting to participate in the study. You can now proceed with the annotation task. Please read the instructions carefully before proceeding."
    )
    st.warning("**It may take up to 20 seconds for the images to load.**")

elif st.session_state.consent == "No":
    # hide the rest of the page
    st.error("You have chosen not to participate in the study.")
    record_non_participation()
    st.error(
        "please copy and paste the following code into Prolific to confirm your choice: CAOIUYYS"
    )
    st.stop()
else:
    st.warning("Please provide your consent to proceed.")
    st.stop()


with st.sidebar:
    st.header("Progress")
    done = st.session_state.progress["done"].notnull().sum()
    total = len(st.session_state.progress)
    st.progress(done / total)
    st.write(f"You have annotated {done} out of {total} items.")

    st.markdown("---")
    st.header("Your selections")
    selected_labels = collect_selected_labels()
    if selected_labels:
        st.write(", ".join(selected_labels))
    else:
        st.write("No labels selected yet.")

    st.markdown("---")
    st.header("Quick instructions")
    st.markdown(
        """
        - Select all emotions that you feel are evoked by the image.
        - You can select multiple emotions.
        - If none of the listed emotions apply, you can specify other emotions in the text boxes.
        - If you feel that the image does not evoke any emotion, select "No emotion".
        - Once you are satisfied with your selections, click the "Confirm" button to proceed to the next image.
        """
    )

st.header("Instructions")
expander = st.expander("Instructions", expanded=True, icon="❗️")
expander.markdown(
    """
    Please read the following instructions carefully before proceeding with the annotation task.
    
    You will be shown a series of images extracted from tweets on X linked to misinformation. **Your task is to identify the emotions each image evokes.**
    This is a subjective task, and there are no right or wrong answers. We are interested in your personal emotional response to each image.
    
    
    
    The possible emotions are categorized into positive and negative emotions. You can select multiple emotions for each image if you feel that more than one emotion is present.
    In particular, you will be asked to identify the presence of the following emotions:
    - **Positive Emotions**: Hope, Joy, Pride, Curiosity
    - **Negative Emotions**: Fear, Anger, Sadness, Ridicule
    - **No Emotion**: If you believe that the image does not evoke any emotion.
    
    Additionally, there are text boxes where you can specify any other positive or negative emotions that you feel are relevant but not listed above.
    
    **Occasionally, images may contain text.** In such cases, please focus on the overall emotional impact of both the image and textual content.

    """
)

notes = load_notes()
st.session_state.progress = get_worker_session(st.session_state.worker_id, notes=notes)
images = loading_images(st.session_state.progress["image_name"].tolist())
next_item_id = select_next_item_for_worker_id(st.session_state.progress)

if next_item_id is None:
    st.success("You have completed all your annotations. Thank you!")
    st.success(
        "Copy and paste the following code into Prolific to receive credit: CGDTTW1Q"
    )
    st.stop()

note = notes.loc[next_item_id]
# image_path = os.path.join(IMAGE_FOLDER, note["image_name"])
# st.write(f"Note loaded in {timeit(time_start)} ms")

image_data = images[note["image_name"]]


item_number = get_item_number(progress=st.session_state.progress)

st.header(f"Annotating item {item_number} out of {len(st.session_state.progress)}")


container = st.container(
    height=650,
    horizontal_alignment="center",
    horizontal=True,
)
with container:
    image_col, annotation_col = st.columns([3, 2])
    with image_col:
        st.image(image_data, caption="Image to annotate")

    with annotation_col:
        col1, col2 = st.columns(2)
        with col1:
            st.header("Positive Emotions")
            for emotion in POSITIVE_EMOTIONS:
                st.checkbox(emotion.capitalize(), key=emotion)

            st.text_input(
                "Other positive emotions (comma separated)", key="other_positive"
            )

        with col2:
            st.header("Negative Emotions")
            for emotion in NEGATIVE_EMOTIONS:
                st.checkbox(emotion.capitalize(), key=emotion)
            st.text_input(
                "Other negative emotions (comma separated)", key="other_negative"
            )

        st.header("No emotion")
        st.checkbox("No emotion", key="none")


st.button(
    "Confirm",
    on_click=lambda: confirm_label(note=note),
    key="confirm_button",
    disabled=not collect_selected_labels(),
)
