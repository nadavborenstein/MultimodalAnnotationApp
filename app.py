import streamlit as st
import random
import os
import pandas as pd
from glob import glob
from st_files_connection import FilesConnection

conn = st.connection("gcs", type=FilesConnection)
df = conn.read("annotation-experiment/myfile.csv", input_format="csv", ttl=600)

for row in df.itertuples():
    st.write(f"{row.Owner} has a :{row.Pet}:")

NOTES = "data/tweets_with_images.csv"
MAX_ANNOTATIONS_PER_WORKER = 50
ID_COL = "tweetId"
IMAGE_FOLDER = "static/images/"


@st.cache_resource
def load_notes() -> pd.DataFrame:
    data = pd.read_csv(NOTES)
    images = glob(os.path.join(IMAGE_FOLDER, "*.png"))
    image_names = [os.path.basename(img) for img in images]
    data = data[data["image_name"].isin(image_names)]
    data = data.drop_duplicates(subset=["image_name"])
    return data


def load_done() -> set:
    done = open("data/done.txt").read()
    done = set(done.split("\n"))
    done = {d.strip() for d in done if d}
    return done


def get_worker_session(worker_id: str):
    # check if a progress file exists for this worker
    progress_file = f"data/worker_progress/progress_{worker_id}.csv"
    if os.path.exists(progress_file):
        progress = pd.read_csv(progress_file)
        return progress
    else:
        notes = load_notes()
        done_notes = load_done()
        notes = notes[~notes[ID_COL].isin(done_notes)]
        notes_to_label = notes.sample(
            n=min(MAX_ANNOTATIONS_PER_WORKER, len(notes)), random_state=42
        )
        ids_to_label = notes_to_label[ID_COL].tolist()
        progress = pd.DataFrame(
            {
                "worker_id": [worker_id] * len(ids_to_label),
                ID_COL: ids_to_label,
                "done": [None] * len(ids_to_label),
                "label": [None] * len(ids_to_label),
            }
        )
        progress.to_csv(progress_file, index=False)
        return progress


def select_next_item_for_worker_id(progress: pd.DataFrame) -> str:
    # Dummy implementation for example purposes
    not_done = progress[progress["done"].isnull()]
    if not_done.empty:
        return None
    next_id = not_done.iloc[0][ID_COL]
    return next_id


def save_label(progress: pd.DataFrame, note: pd.Series):
    progress_file = f"data/worker_progress/progress_{st.session_state.worker_id}.csv"
    index = progress[progress[ID_COL] == note[ID_COL]].index
    if index.empty:
        st.error("Error: Note ID not found in progress.")
        return
    index = index[0]
    progress.at[index, "done"] = True
    progress.at[index, "label"] = str(st.session_state.emotion_label)
    progress.to_csv(progress_file, index=False)


st.title("Annotation experiment")

with st.container(key="consent_container"):
    st.header("Consent")
    st.radio(
        label="Do you consent to participate in this study?",
        options=["", "No", "Yes"],
        key="consent",
        horizontal=True,
        index=0,
        help="You must consent to participate in this study to proceed.",
        on_change=lambda: st.session_state.update({"show_consent": False}),
    )


if st.session_state.consent == "Yes":
    st.session_state.show_consent = False
    st.success("Thank you for consenting to participate in the study.")
    st.write("You can now proceed with the annotation task.")
    # Here you can add the code to display the annotation task
elif st.session_state.consent == "No":
    # hide the rest of the page
    st.error("You have chosen not to participate in the study.")
    st.text_input(
        "Please enter your Prolific ID to confirm your choice",
        key="worker_id",
        placeholder="ID",
        value=st.session_state.get("worker_id", ""),
    )
    st.stop()
else:
    st.warning("Please provide your consent to proceed.")
    st.stop()

if "worker_id" not in st.session_state:
    placeholder = "ID"
else:
    placeholder = st.session_state.worker_id

st.text_input(
    "Please enter your Prolific ID",
    key="worker_id",
    placeholder=placeholder,
    value=st.session_state.get("worker_id", ""),
)
if not st.session_state.worker_id:
    st.warning("Please enter your Prolific ID to proceed.")
    st.stop()
else:
    st.success(
        f"Thank you for providing your Prolific ID: {st.session_state.worker_id}."
    )

with st.expander("Instructions", expanded=True):
    st.markdown(
        """
    Please read the following instructions carefully before proceeding with the annotation task.

    1. You will be presented with a series of images and corresponding text snippets.
    2. Your task is to evaluate whether the text accurately describes the content of the image.
    3. For each image-text pair, select one of the following options:
       - "Accurate": The text accurately describes the image.
       - "Inaccurate": The text does not accurately describe the image.
       - "Unclear": It is unclear whether the text describes the image accurately.
    4. Please take your time to consider each pair carefully before making your selection.
    5. If you have any questions or need clarification, please contact the study administrator.

    Thank you for your participation!
    """
    )


notes = load_notes()
progress = get_worker_session(st.session_state.worker_id)
next_item_id = select_next_item_for_worker_id(progress)
if next_item_id is None:
    st.success("You have completed all your annotations. Thank you!")
    st.stop()

note = notes[notes[ID_COL] == next_item_id].iloc[0]
image_path = os.path.join(IMAGE_FOLDER, note["image_name"])
st.image(image_path, caption="Image to annotate", use_container_width=True)

st.multiselect(
    "What emotion does the image evoke?",
    key="emotion_label",
    options=["None", "Fear", "Anger", "Hope", "Joy"],
)
st.button("Confirm", on_click=lambda: save_label(progress=progress, note=note))
