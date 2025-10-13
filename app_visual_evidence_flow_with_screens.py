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
import pickle
from typing import Tuple, List, Dict

st.set_page_config(layout="wide")
conn = st.connection("gcs", type=FilesConnection)

LANGUAGE = "en"
TASK_NAME = f"visual_evidence_head_{LANGUAGE}"
NOTES = "annotation-experiment/data/multimodal_tweets_balanced.csv"
DEEPEST_NODE = 6

DONE_CODE = "CV8TK0ZL"
DONE_LINK = f"https://app.prolific.com/submissions/complete?cc={DONE_CODE}"
NO_CONCENT_CODE = "C1B7DNHB"
NO_CONCENT_LINK = f"https://app.prolific.com/submissions/complete?cc={NO_CONCENT_CODE}"

SCREENED_CODE = "MEEP"
SCREENED_LINK = f"https://app.prolific.com/submissions/complete?cc={SCREENED_CODE}"

ADD_QUALIFICATIONS = True
QUALIFICATION_NOTES = "annotation-experiment/data/en_qualification_data.csv"
INSTRUCTIONS_FILE = "static/instructions.txt"
QUALIFICATION_IMAGE_FOLDER = "annotation-experiment/static/qualification_images/"
QUESTION_TREE = "static/question_tree.yaml"
MAX_ANNOTATIONS_PER_WORKER = 25  # TODO: adjust as needed
ID_COL = "id_str"
IMAGE_FOLDER = "annotation-experiment/static/resized_images/"
PROGRESS_FOLDER = f"annotation-experiment/data/worker_progress/{TASK_NAME}"
DONE_FILE = f"annotation-experiment/data/done_{TASK_NAME}.txt"
NON_PARTICIPANTS_FILE = "annotation-experiment/data/non_participants.txt"
NUM_ANNOTATORS_PER_ITEM = 6  # TODO: adjust as needed


DEBUGGING = True
NUM_NOTES_IN_DEBUGGING = MAX_ANNOTATIONS_PER_WORKER

INSTRUCTIONS = open(INSTRUCTIONS_FILE, "r").read()
INSTRUCTIONS = INSTRUCTIONS.replace(
    "NUM_QUESTIONS",
    (
        str(MAX_ANNOTATIONS_PER_WORKER + 5)
        if ADD_QUALIFICATIONS
        else str(MAX_ANNOTATIONS_PER_WORKER)
    ),
)
QUALIFICATION_GT: List[Tuple[str, str, str]] = pickle.load(
    open("static/qualifications_gt.pkl", "rb")
)


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


def is_disqualified() -> bool:
    if "qualification_status" in st.session_state:
        return st.session_state.qualification_status
    progress = st.session_state.progress
    qualifications = progress[progress["qualification"]]
    all_done = qualifications["done"].notnull().all()

    # Even if not all qualifications are done, if any answer is wrong, the worker is disqualified
    for qualification_id, QAs in QUALIFICATION_GT:

        worker_annotations = qualifications[
            qualifications[ID_COL] == int(qualification_id)
        ].iloc[0]
        if pd.isna(worker_annotations["done"]):
            continue

        worker_answers = eval(worker_annotations["label"])
        worker_answers = [(q, a) for q, a, _ in worker_answers]
        for q, a in QAs:
            if (q, a) not in worker_answers:
                st.session_state.qualification_status = True
                return True

    # if all qualifications are done and all answers are correct, the worker is qualified
    if all_done:
        st.session_state.qualification_status = False
    return False


@st.cache_data
def anonimize_links(text: str) -> str:
    # find all links in the text
    urls = re.findall(r"http\S+|www\S+|https\S+", text, re.IGNORECASE)
    for root_url in urls:
        url = root_url.strip().replace("http://", "").replace("https://", "")
        top_url = url[: url.find("/")] if "/" in url else url
        the_rest = url[len(top_url) :]
        anonimized_url = "www." + top_url + the_rest[:10] + "..."
        text = text.replace(root_url, anonimized_url)
    return text


def remove_links(text: str):
    urls = re.findall(r"http\S+|www\S+|https\S+", text, re.IGNORECASE)
    for url in urls:
        text = text.replace(url, "")
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
    file = open(QUESTION_TREE, "r")
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
    st.write(notes.shape)
    notes = notes[notes["language_present"] == LANGUAGE]
    st.write(notes.shape)
    # seed from worker_id
    images = conn.fs.glob(f"{IMAGE_FOLDER}*.jpeg")
    image_names = [os.path.basename(img) for img in images]
    notes = notes[notes["image_name"].isin(image_names)]
    st.write(notes.shape)
    notes = notes.drop_duplicates(subset=["image_name"])
    st.write(notes.shape)
    if DEBUGGING:
        notes = notes.head(NUM_NOTES_IN_DEBUGGING)
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
        st.write(notes.shape)
        st.write(notes.head())
        if ADD_QUALIFICATIONS:
            qualifications = notes[notes["qualification"]]
            non_qualifications = notes[~notes["qualification"]].sample(
                n=min(MAX_ANNOTATIONS_PER_WORKER, len(notes)), random_state=seed
            )
            notes_to_label = pd.concat([qualifications, non_qualifications])
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
                "qualification": notes_to_label["qualification"].tolist(),
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
        for type in ["image", "text", "claim"]:
            for suffix in ["", "_text", "_confirm"]:
                key = f"{type}_question_{i}{suffix}"
                if key in st.session_state:
                    del st.session_state[key]

    if "has_claim" in st.session_state:
        del st.session_state["has_claim"]
    if "has_claim_text" in st.session_state:
        del st.session_state["has_claim_text"]
    if "has_claim_confirm" in st.session_state:
        del st.session_state["has_claim_confirm"]
    st.session_state.labels = []


def collect_selected_labels() -> list:
    """
    Collect selected labels from the session state.
    Returns a list of selected labels.
    """
    labels = []
    if "has_claim" in st.session_state and st.session_state.has_claim == "No":
        labels.append(
            (
                "It is impossible to annotate this image",
                st.session_state.has_claim,
                st.session_state.has_claim_text,
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
    possible_answers = [s.capitalize() for s in possible_answers]
    possible_answers.sort(reverse=True)
    possible_next_questions = current_question["answers"]
    return question, possible_answers, possible_next_questions


def save_value(question, key):
    if "labels" not in st.session_state:
        st.session_state.labels = []
    multi_choice_answer = st.session_state[key]
    free_text_answer = st.session_state[f"{key}_text"]
    st.session_state.labels.append((question, multi_choice_answer, free_text_answer))
    st.session_state.question_counter += 1


def is_mandatory_text(current_question):
    is_mandatory = current_question["mandatory_text"]
    if is_mandatory == False:
        return "None"
    return is_mandatory.split("-")[1]


def is_multi_answers(current_question):
    if "multiple_answers" in current_question:
        return current_question["multiple_answers"]
    return False


def disable_confirm(mandatory_text, ans, text_ans):
    if not ans:
        return True

    if mandatory_text and not text_ans:
        return True

    return False


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
    st.header("Quick instructions")
    st.markdown(
        """
        - Read the tweet text and examine the image carefully.
        - Read the additional context provided to understand why the tweet/image was flagged as misinformation.
        - Determine whether the tweet/image contain an explicit or implicit claim
        - Answer the questions to the best of your ability.
        - If a free-text input is mandatory, it will be marked as so.
        - Click "Confirm" to save your annotations and proceed to the next item.
        """
    )

with st.spinner("**Loading images...**", show_time=True):
    images = load_images(st.session_state.progress["image_name"].tolist())
    next_item_id = select_next_item_for_worker_id(st.session_state.progress)


if is_disqualified():
    st.success(
        "Your responses didn't meet the criteria we are looking for in our study. You will still be paid for your time.."
    )
    st.success(
        f"Click on the link below or copy and paste the following code into Prolific to receive payment: {SCREENED_CODE}"
    )
    st.link_button(
        "back to Prolific",
        SCREENED_LINK,
        type="primary",
    )
    st.stop()


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
tweet_text = remove_links(note.full_text)

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
        st.subheader("Tweet image 🖼️")
        with st.container(border=True):
            st.image(image_data)
    with text_col:
        st.subheader("Tweet text 💬")
        st.markdown(
            f'<div dir="auto">{tweet_text}</div>',
            unsafe_allow_html=True,
        )
        # st.write(tweet_text)
        st.markdown("---")
        with st.container(border=False):
            title = "Additional context 💡"
            st.markdown(
                f'<div style="background-color:#E8F6FF;Height:auto" dir="auto"><h3>{title}</h3>{note_text}</div>',
                unsafe_allow_html=True,
            )

if "question_counter" not in st.session_state:
    st.session_state.question_counter = 1

st.divider()
claim = True
# not a claim
placeholder = st.empty()
with placeholder.container():
    st.markdown(
        f"**Does the tweet and/or image make a claim? (either explicitly or implicitly)**"
    )
    st.pills(
        "**Claim**: A statement that asserts something about reality, which can, in principle, be evaluated as true or false.\n\n**Remember:** the claim can be implicit, for example, sharing a fake image with the Tweet text implicitly claiming that the image is real (e.g., 'Look at this! It is terrible!').",
        ["Yes", "No"],
        selection_mode="single",
        key="has_claim",
        default=None,
    )
    st.text_input(
        "If not, explain why. If yes, describe the claim in your own words (**required**)",
        key=f"has_claim_text",
        placeholder="",
        value=st.session_state.get(f"has_claim_text", ""),
        help="Please explain your choice in a few words.",
    )
    st.checkbox(
        label="Confirm",
        value=False,
        key=f"has_claim_confirm",
        disabled=not st.session_state["has_claim_text"],
    )
    if not st.session_state["has_claim_confirm"]:
        st.stop()
    elif st.session_state["has_claim"] == "No":
        confirm_label(note=note)
        st.session_state["has_claim"] = None
        st.session_state["has_claim_text"] = ""
        st.session_state["has_claim_confirm"] = False
        st.rerun()
    placeholder.empty()

placeholder.empty()
current_question = question_tree["claim_identification"]
placeholder = st.empty()

with placeholder:
    for i in range(DEEPEST_NODE):
        question, possible_answers, possible_next_questions = get_question(
            current_question
        )
        mandatory_text_answer: str = is_mandatory_text(current_question)
        multi_answers = is_multi_answers(current_question)
        explanation = current_question["explanation"]

        with st.container():
            st.subheader(f"{st.session_state.question_counter}) Text related questions")
            st.markdown(f"**{question}**")
            st.pills(
                explanation,
                possible_answers,
                selection_mode="multi" if multi_answers else "single",
                key=f"claim_question_{i}",
                default=None,
                args=[f"claim_question_{i}", question],
            )
            if mandatory_text_answer != "None":
                mandatory_text = True
                text_input_title = "Explain your choice **(required)**"
            else:
                mandatory_text = False
                text_input_title = "Explain your choice (optional)"

            st.text_input(
                text_input_title,
                key=f"claim_question_{i}_text",
                placeholder="",
                value=st.session_state.get(f"claim_question_{i}_text", ""),
                disabled=not st.session_state[f"claim_question_{i}"],
                help="Please explain your choice in a few words.",
                args=[f"claim_question_{i}_text", "Explain your choice"],
            )
            st.checkbox(
                label="Confirm",
                value=False,
                key=f"claim_question_{i}_confirm",
                disabled=disable_confirm(
                    mandatory_text,
                    st.session_state[f"claim_question_{i}"],
                    st.session_state[f"claim_question_{i}_text"],
                ),
                on_change=save_value,
                args=[question, f"claim_question_{i}"],
            )
        if not st.session_state[f"claim_question_{i}_confirm"]:
            st.stop()
        if multi_answers:
            break

        answer = st.session_state[f"claim_question_{i}"]
        current_question = possible_next_questions.get(answer)
        if "label" in current_question:
            break

placeholder.empty()
current_question = question_tree["image"]
# image related stuff
placeholder = st.empty()

with placeholder:
    for i in range(DEEPEST_NODE):
        question, possible_answers, possible_next_questions = get_question(
            current_question
        )
        mandatory_text_answer: str = is_mandatory_text(current_question)
        multi_answers = is_multi_answers(current_question)
        explanation = current_question["explanation"]

        with st.container():
            st.subheader(
                f"{st.session_state.question_counter}) Image related questions"
            )
            st.markdown(f"**{question}**")
            st.pills(
                explanation,
                possible_answers,
                selection_mode="multi" if multi_answers else "single",
                key=f"image_question_{i}",
                default=None,
                args=[f"image_question_{i}", question],
            )

            if mandatory_text_answer != "None":
                mandatory_text = True
                text_input_title = "Explain your choice **(required)**"
            else:
                mandatory_text = False
                text_input_title = "Explain your choice (optional)"

            st.text_input(
                text_input_title,
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
                disabled=disable_confirm(
                    mandatory_text,
                    st.session_state[f"image_question_{i}"],
                    st.session_state[f"image_question_{i}_text"],
                ),
                on_change=save_value,
                args=[question, f"image_question_{i}"],
            )
        if not st.session_state[f"image_question_{i}_confirm"]:
            st.stop()
        if multi_answers:
            break
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
        mandatory_text_answer: str = is_mandatory_text(current_question)
        multi_answers = is_multi_answers(current_question)
        explanation = current_question["explanation"]

        with st.container():
            st.subheader(f"{st.session_state.question_counter}) Text related questions")
            st.markdown(f"**{question}**")
            st.pills(
                explanation,
                possible_answers,
                selection_mode="multi" if multi_answers else "single",
                key=f"text_question_{i}",
                default=None,
                args=[f"text_question_{i}", question],
            )
            if mandatory_text_answer != "None":
                mandatory_text = True
                text_input_title = "Explain your choice **(required)**"
            else:
                mandatory_text = False
                text_input_title = "Explain your choice (optional)"

            st.text_input(
                text_input_title,
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
                disabled=disable_confirm(
                    mandatory_text,
                    st.session_state[f"text_question_{i}"],
                    st.session_state[f"text_question_{i}_text"],
                ),
                on_change=save_value,
                args=[question, f"text_question_{i}"],
            )
        if not st.session_state[f"text_question_{i}_confirm"]:
            st.stop()
        if multi_answers:
            break

        answer = st.session_state[f"text_question_{i}"]
        current_question = possible_next_questions.get(answer)
        if "label" in current_question:
            break


st.info("loading next image")
st.session_state.question_counter = 1
confirm_label(note=note)
st.rerun()
