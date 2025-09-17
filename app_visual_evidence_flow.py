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
import re
import yaml
import time

st.set_page_config(layout="wide")
conn = st.connection("gcs", type=FilesConnection)

LANGUAGE = "en"
TASK_NAME = f"visual_evidence_head_{LANGUAGE}"
NOTES = "annotation-experiment/data/multimodal_tweets_balanced.csv"
DEEPEST_NODE = 4

DONE_CODE = "CV8TK0ZL"
DONE_LINK = f"https://app.prolific.com/submissions/complete?cc={DONE_CODE}"
NO_CONCENT_CODE = "C1B7DNHB"
NO_CONCENT_LINK = f"https://app.prolific.com/submissions/complete?cc={NO_CONCENT_CODE}"


ADD_QUALIFICATIONS = True
QUALIFICATION_NOTES = "annotation-experiment/data/en_qualification_data.csv"
QUALIFICATION_IMAGE_FOLDER = "annotation-experiment/static/qualification_images/"
QUESTION_TREE = "annotation-experiment/static/question_tree.yaml"
MAX_ANNOTATIONS_PER_WORKER = 25  # TODO: adjust as needed
ID_COL = "tweet_id"
IMAGE_FOLDER = "annotation-experiment/static/resized_images/"
PROGRESS_FOLDER = f"annotation-experiment/data/worker_progress/{TASK_NAME}"
DONE_FILE = f"annotation-experiment/data/done_{TASK_NAME}.txt"
NON_PARTICIPANTS_FILE = "annotation-experiment/data/non_participants.txt"
NUM_ANNOTATORS_PER_ITEM = 3  # TODO: adjust as needed


DEBUGGING = True

INSTRUCTIONS = """
    """


def time_before():
    return int(time() * 1000)


def my_badge(text, colour) -> str:
    return f":{colour}-badge[{text}]"


def timeit(start_time):
    return int(time() * 1000) - start_time


def append_to_file(item: str, file_path: str):
    done = conn.fs.open(file_path, "r").read()
    done += f"{item}\n"
    conn.fs.open(file_path, "w").write(done)


@st.cache_data
def anonimize_links(text: str) -> str:
    # find all links in the text
    urls = re.findall(r"http\S+|www\S+|https\S+", text, re.IGNORECASE)
    for root_url in urls:
        url = root_url.strip().replace("http://", "").replace("https://", "")
        top_url = url[: url.find("/")] if "/" in url else url
        anonimized_url = "www." + top_url + "/[LINK]"
        text = text.replace(root_url, anonimized_url)
    return text


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
def load_question_tree() -> dict:
    file = conn.fs.open(QUESTION_TREE, "r")
    question_tree = yaml.safe_load(file)

    # replace boolean keys with "yes" and "no"
    def replace_bool_keys(d):
        if isinstance(d, dict):
            new_dict = {}
            for k, v in d.items():
                if k is True:
                    k = "Yes"
                elif k is False:
                    k = "No"
                new_dict[k] = replace_bool_keys(v)
            return new_dict
        elif isinstance(d, list):
            return [replace_bool_keys(i) for i in d]
        else:
            return d

    question_tree = replace_bool_keys(question_tree)
    return question_tree


@st.cache_resource
def load_qualification_notes() -> pd.DataFrame:
    notes = conn.fs.open(QUALIFICATION_NOTES, "r").read()
    notes = pd.read_csv(io.StringIO(notes))
    images = conn.fs.glob(f"{QUALIFICATION_IMAGE_FOLDER}*.jpeg")
    image_names = [os.path.basename(img) for img in images]
    notes = notes[notes["image_name"].isin(image_names)]
    notes = notes.drop_duplicates(subset=["image_name"])
    notes.set_index(ID_COL, inplace=True, drop=False)
    return notes


@st.cache_resource
def load_notes() -> pd.DataFrame:
    notes = conn.fs.open(NOTES, "r").read()
    notes = pd.read_csv(io.StringIO(notes))
    notes = notes[notes["language_present"] == LANGUAGE]
    # seed from worker_id
    images = conn.fs.glob(f"{IMAGE_FOLDER}*.jpeg")
    image_names = [os.path.basename(img) for img in images]
    notes = notes[notes["image_name"].isin(image_names)]
    notes = notes.drop_duplicates(subset=["image_name"])
    if DEBUGGING:
        notes = notes.head(25)
    notes.set_index(ID_COL, inplace=True, drop=False)

    if ADD_QUALIFICATIONS:
        qualification_notes = load_qualification_notes()
        notes = pd.concat([notes, qualification_notes])
        notes["qualification"] = notes.index.isin(qualification_notes.index)
    return notes


def load_done() -> set:
    if not conn.fs.exists(DONE_FILE):
        conn.fs.open(DONE_FILE, "w").write("")
        return set()

    done = conn.fs.open(DONE_FILE, "r").read()
    done = done.split("\n")
    done = [d.strip() for d in done if d]
    counts = Counter(done)
    done = {d for d, c in counts.items() if c >= NUM_ANNOTATORS_PER_ITEM}

    return done


@st.cache_data
def load_images(image_names) -> list:
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

        if ADD_QUALIFICATIONS:
            qualifications = notes[notes["qualification"]]
            non_qualifications = notes[~notes["qualification"]].sample(
                n=min(MAX_ANNOTATIONS_PER_WORKER, len(notes)), random_state=seed
            )
            notes_to_label = pd.concat([qualifications, non_qualifications])
            notes_to_label = notes_to_label.sample(frac=1, random_state=seed)
        else:
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
    for i in range(DEEPEST_NODE):
        for type in ["image", "text"]:
            for suffix in ["", "_text", "_confirm"]:
                key = f"{type}_question_{i}{suffix}"
                if key in st.session_state:
                    del st.session_state[key]

    if "cannot_annotate" in st.session_state:
        del st.session_state["cannot_annotate"]
    if "cannot_annotate_text" in st.session_state:
        del st.session_state["cannot_annotate_text"]
    st.session_state.labels = []


def collect_selected_labels() -> list:
    """
    Collect selected labels from the session state.
    Returns a list of selected labels.
    """
    labels = []
    if "cannot_annotate" in st.session_state and st.session_state.cannot_annotate:
        labels.append(
            (
                "It is impossible to annotate this image",
                st.session_state.cannot_annotate_text,
            )
        )
    if "labels" in st.session_state:
        labels.extend(st.session_state.labels)
    return labels


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


@st.cache_data
def get_question(current_question):
    question = current_question["question"]
    possible_answers = current_question["answers"].keys()
    possible_answers = list(possible_answers)
    possible_answers.sort()
    possible_next_questions = current_question["answers"]
    return question, possible_answers, possible_next_questions


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
st.pills(
    label="Do you consent to participate in this study?",
    options=["No", "Yes"],
    key="consent",
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
    st.warning(
        "**It may take up to 20 seconds for the images to load. Please carefuly read the instructions in the meanwhile.**"
    )

elif st.session_state.consent == "No":
    # hide the rest of the page
    st.error("You have chosen not to participate in the study.")
    record_non_participation()
    st.error(
        f"Click on the link below or copy and paste the following code into Prolific to confirm your choice: {NO_CONCENT_CODE}"
    )
    st.link_button(
        "Back to Prolific",
        NO_CONCENT_LINK,
        type="primary",
    )
    st.stop()
else:
    st.warning("Please provide your consent to proceed.")
    st.stop()


st.header("Instructions")
expander = st.expander("Instructions", expanded=True, icon="❗️")
expander.markdown(INSTRUCTIONS)

with st.spinner("Loading your annotation session...", show_time=True):
    notes = load_notes()
    question_tree = load_question_tree()
    st.session_state.question_tree = question_tree
    if "current_question" not in st.session_state:
        st.session_state.current_question = question_tree["image"]

    st.session_state.progress = get_worker_session(
        st.session_state.worker_id, notes=notes
    )

with st.sidebar:
    st.header("Progress")
    done = st.session_state.progress["done"].notnull().sum()
    total = len(st.session_state.progress)
    st.progress(done / total)
    st.write(f"You have annotated {done} out of {total} items.")

    st.markdown("---")
    st.header("Your selections")
    selected_labels = collect_selected_labels()
    badges = []
    # if "cannot_annotate" in st.session_state and st.session_state.cannot_annotate:
    #     badges.append(my_badge("cannot annotate", "grey"))
    # if (
    #     "real_image" in st.session_state
    #     and QUESTION_OPTIONS["real_image"][1] == st.session_state.real_image
    # ):
    #     badges.append(my_badge("non genuine image", "red"))
    # if (
    #     "real_source" in st.session_state
    #     and QUESTION_OPTIONS["real_source"][1] == st.session_state.real_source
    # ):
    #     badges.append(my_badge("unreliable source", "green"))
    # if (
    #     "tweet_text" in st.session_state
    #     and QUESTION_OPTIONS["tweet_text"][1] == st.session_state.tweet_text
    # ):
    #     badges.append(my_badge("misleading tweet text", "blue"))
    # if (
    #     "embedded_text" in st.session_state
    #     and QUESTION_OPTIONS["embedded_text"][2] == st.session_state.embedded_text
    # ):
    #     badges.append(my_badge("misleading image text", "violet"))

    # if badges:
    #     st.markdown(" ".join(badges))

    st.markdown("---")
    st.header("Quick instructions")
    st.markdown(
        """
        - Read the tweet text and examine the image carefully.
        - Read the additional context provided to understand why the tweet/image was flagged as misinformation.
        - Answer the questions to the best of your ability.
        - If you are unsure about a question, select "I don't know/not relevant".
        - If the tweet/image cannot be annotated, select "It is impossible to annotate this image" and briefly explain why.
        - Click "Confirm" to save your annotations and proceed to the next item.
        """
    )

with st.spinner("**Loading images...**", show_time=True):
    images = load_images(st.session_state.progress["image_name"].tolist())
    next_item_id = select_next_item_for_worker_id(st.session_state.progress)

if next_item_id is None:
    st.success("You have completed all your annotations. Thank you!")
    st.success(
        f"Click on the link below or copy and paste the following code into Prolific to receive credit: {DONE_CODE}"
    )
    st.link_button(
        "back to Prolific",
        DONE_LINK,
        type="primary",
    )
    st.stop()

note = notes.loc[next_item_id]


image_data = images[note["image_name"]]
note_text = anonimize_links(note.note)
tweet_text = anonimize_links(note.full_text)

item_number = get_item_number(progress=st.session_state.progress)

st.header(f"Annotating item {item_number} out of {len(st.session_state.progress)}")


container = st.container(
    horizontal_alignment="center",
    horizontal=True,
    border=True,
)
with container:
    image_col, text_col = st.columns([3, 2])
    with image_col:
        st.subheader("Tweet image")
        st.image(image_data)
    with text_col:
        st.subheader("Tweet text")
        st.markdown(
            f'<div dir="auto">{tweet_text}</div>',
            unsafe_allow_html=True,
        )
        # st.write(tweet_text)
        st.markdown("---")
        st.subheader("Additional context")
        st.markdown(
            f'<div dir="auto">{note_text}</div>',
            unsafe_allow_html=True,
        )


st.divider()
# st.session_state.labels = []
# if "counter" not in st.session_state:
#     st.session_state.counter = 0

# # not a claim
# placeholder = st.empty()
# with placeholder:
#     with st.container():
#         st.markdown(f"**Is there a claim?**")
#         st.pills(
#             "Select an answer:",
#             ["Yes", "No"],
#             selection_mode="single",
#             key="has_claim",
#             default=None,
#         )
#     if not st.session_state["has_claim"]:
#         st.stop()
#     elif st.session_state["has_claim"] == "No":
#         confirm_label(note=note)
#         st.rerun()
#     else:
#         placeholder.empty()


def save_value(question, key):
    if "labels" not in st.session_state:
        st.session_state.labels = []
    multi_choice_answer = st.session_state[key]
    free_text_answer = st.session_state[f"{key}_text"]
    st.session_state.labels.append((question, multi_choice_answer, free_text_answer))


current_question = question_tree["image"]
# image related stuff
placeholder = st.empty()
with placeholder:
    for i in range(DEEPEST_NODE):
        question, possible_answers, possible_next_questions = get_question(
            current_question
        )
        with st.container():
            st.subheader("Image related questions")
            st.markdown(f"**{question}**")
            st.pills(
                "Select an answer:",
                possible_answers,
                selection_mode="single",
                key=f"image_question_{i}",
                default=None,
                args=[f"image_question_{i}", question],
            )
            st.text_input(
                "Explain your choice",
                key=f"image_question_{i}_text",
                placeholder="",
                value=st.session_state.get(f"image_question_{i}_text", ""),
                disabled=not st.session_state[f"image_question_{i}"],
                help="Please explain your choice in a few words.",
                args=[f"image_question_{i}_text", "Explain your choice"],
            )
            st.checkbox(
                label="Confirm",
                value=False,
                key=f"image_question_{i}_confirm",
                disabled=not st.session_state[f"image_question_{i}_text"],
                on_change=save_value,
                args=[question, f"image_question_{i}"],
            )
        if not st.session_state[f"image_question_{i}_confirm"]:
            st.stop()

        answer = st.session_state[f"image_question_{i}"]
        current_question = possible_next_questions.get(answer)
        if "label" in current_question:
            break

placeholder.empty()
current_question = question_tree["text"]
placeholder = st.empty()

with placeholder:
    for i in range(DEEPEST_NODE):
        question, possible_answers, possible_next_questions = get_question(
            current_question
        )
        with st.container():
            st.subheader("Text related questions")
            st.markdown(f"**{question}**")
            st.pills(
                "Select an answer:",
                possible_answers,
                selection_mode="single",
                key=f"text_question_{i}",
                default=None,
                args=[f"text_question_{i}", question],
            )
            st.text_input(
                "Explain your choice",
                key=f"text_question_{i}_text",
                placeholder="",
                value=st.session_state.get(f"text_question_{i}_text", ""),
                disabled=not st.session_state[f"text_question_{i}"],
                help="Please explain your choice in a few words.",
                args=[f"text_question_{i}_text", "Explain your choice"],
            )
            st.checkbox(
                label="Confirm",
                value=False,
                key=f"text_question_{i}_confirm",
                disabled=not st.session_state[f"text_question_{i}_text"],
                on_change=save_value,
                args=[question, f"text_question_{i}"],
            )
        if not st.session_state[f"text_question_{i}_confirm"]:
            st.stop()

        answer = st.session_state[f"text_question_{i}"]
        current_question = possible_next_questions.get(answer)
        if "label" in current_question:
            break

st.info("loading next image")

confirm_label(note=note)
st.rerun()
