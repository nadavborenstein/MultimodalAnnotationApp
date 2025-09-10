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

st.set_page_config(layout="wide")
conn = st.connection("gcs", type=FilesConnection)

LANGUAGE = "de"
TASK_NAME = f"visual_evidence_head_{LANGUAGE}"
NOTES = "annotation-experiment/data/multimodal_tweets_balanced.csv"
MAX_ANNOTATIONS_PER_WORKER = 25  # TODO: adjust as needed
ID_COL = "tweet_id"
IMAGE_FOLDER = "annotation-experiment/static/resized_images/"
PROGRESS_FOLDER = f"annotation-experiment/data/worker_progress/{TASK_NAME}"
DONE_FILE = f"annotation-experiment/data/done_{TASK_NAME}.txt"
NON_PARTICIPANTS_FILE = "annotation-experiment/data/non_participants.txt"
NUM_ANNOTATORS_PER_ITEM = 3  # TODO: adjust as needed
LABELS = [
    "real_image",
    "real_source",
    "tweet_text",
    "embedded_text",
    "cannot_annotate",
]


LABELS_TEXT = [l + "_text" for l in LABELS]
QUESTION_OPTIONS = {
    "real_image": [
        "The image is genuine",
        "The image is **not** genuine (e.g., edited or AI generated without disclosure)",
        "I don't know/not relevant",
    ],
    "real_source": [
        "The image originate from a reliable and verified source",
        "The image **does not** originate from a reliable, verified, source (imposter, satire, etc.)",
        "I don't know/not relevant",
    ],
    "tweet_text": [
        "The claim in the tweet's text faithfully represent the content of the image",
        "The claim in the tweet's text **does not** faithfully represent the content of the image",
        "I don't know/not relevant",
    ],
    "embedded_text": [
        "There is no textual claim in the image",
        "The claim in the image faithfully represent the visual content of the image",
        "The claim in the image **does not** faithfully represent the visual content of the image",
        "I don't know/not relevant",
    ],
    "cannot_annotate": [
        False,
        True,
    ],
}

DEBUGGING = True

INSTRUCTIONS = """
    Please read the following instructions carefully before proceeding with the annotation task.
    
    We are studying how images on X (formerly Twitter) are used to spread misinformation online.
    Misinformation can have serious consequences, including shaping public opinion, eroding trust in institutions, and even incite violence.
    By understanding how images are used to spread misinformation misinformation, we can develop better strategies to limit its impact.
    
    **In this study we wish to analyse high-level properties images and its relationship to claim in the tweet**. 
    
    You will be given:
    
    * A series of tweets linked to misinformation.
        
    * The images that were attached to the tweets.
        
    * Additional context explaining what is problematic with the tweet and/or image and why it was flagged as misinformation.

    
    **Please take your time to carefully examine the tweet and the image. Understand the claim that is made, and the reason it is misleading. Then, answer the questions below.**
            
    1. **Is the image genuine?** 
    A genuine image is an original image that was not altered without disclosure.
    A **non-genuine** image is a fake (e.g., AI generated, forged document) or altered image (cropped, photoshoped, etc.,) that is not disclosed as such, with the purpose of misleading. If the image is not genuine, please explain how.
    
    2. **Does the image originate from a reliable and verified source?** 
    Here we refer specifically to the original creator of the image content. A source is considered unreliable if it lacks credibility or authority, such as impersonators or satirical platforms. If the source is not reliable, please explain how.
    
    3. **Does the claim in the tweet's text faithfully represent the content of the image?** 
    Read the tweet text and evaluate whether the claim made in the tweet accurately reflects the content of the attached image. If it does not, please explain how.
    
    4. **If the image contains a textual claim, does the claim faithfully represent the visual content of the image?**
    If the image contains text (e.g., a screenshot of a tweet or news article, a meme, a document, etc.), read the text in the image and evaluate whether the claim made in the text accurately reflects the visual content of the image. If it does not, please explain how.
    
    **Please answer the questions to the best of your ability.** If you are unsure about a question, or a question cannot be answered, select “I don't know/not relevant”.
    If a tweet or image cannot be annotated (e.g., it is not misleading, or the claim is unclear), select “It is impossible to annotate this image” and briefly explain why.     
    
    **Remember, You can opt out of this study at any time with no negative consequences.**
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
    for label in LABELS:
        if label in st.session_state:
            st.session_state[label] = QUESTION_OPTIONS[label][0]
    for label_text in LABELS_TEXT:
        if label_text in st.session_state:
            st.session_state[label_text] = ""


def collect_selected_labels() -> list:
    """
    Collect selected labels from the session state.
    Returns a list of selected labels.
    """
    selected_labels = []
    for label in LABELS:
        if label in st.session_state:
            selected_labels.append(
                f"question: {label}. answer: {st.session_state[label]}"
            )

    for label_text in LABELS_TEXT:
        if label_text in st.session_state and st.session_state[label_text].strip():
            selected_labels.append(
                f"question: {label_text}. answer: {st.session_state[label_text].strip()}"
            )

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
        "Click on the link below or copy and paste the following code into Prolific to confirm your choice: C1B7DNHB"
    )
    st.link_button(
        "Back to Prolific",
        "https://app.prolific.com/submissions/complete?cc=C1B7DNHB",
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
    if "cannot_annotate" in st.session_state and st.session_state.cannot_annotate:
        badges.append(my_badge("cannot annotate", "grey"))
    if (
        "real_image" in st.session_state
        and QUESTION_OPTIONS["real_image"][1] == st.session_state.real_image
    ):
        badges.append(my_badge("non genuine image", "red"))
    if (
        "real_source" in st.session_state
        and QUESTION_OPTIONS["real_source"][1] == st.session_state.real_source
    ):
        badges.append(my_badge("unreliable source", "green"))
    if (
        "tweet_text" in st.session_state
        and QUESTION_OPTIONS["tweet_text"][1] == st.session_state.tweet_text
    ):
        badges.append(my_badge("misleading tweet text", "blue"))
    if (
        "embedded_text" in st.session_state
        and QUESTION_OPTIONS["embedded_text"][2] == st.session_state.embedded_text
    ):
        badges.append(my_badge("misleading image text", "violet"))

    if badges:
        st.markdown(" ".join(badges))

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
        "Click on the link below or copy and paste the following code into Prolific to receive credit: CV8TK0ZL"
    )
    st.link_button(
        "back to Prolific",
        "https://app.prolific.com/submissions/complete?cc=CV8TK0ZL",
        type="primary",
    )
    st.stop()

note = notes.loc[next_item_id]
# image_path = os.path.join(IMAGE_FOLDER, note["image_name"])
# st.write(f"Note loaded in {timeit(time_start)} ms")

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


image_focused, image_text_focused = st.columns(2)
with image_focused:
    st.write(
        "**Is the image genuine? A non genuine image is a fake or altered image that is not disclosed as such.**"
    )
    st.radio(
        "genuine image",
        QUESTION_OPTIONS["real_image"],
        index=0,
        horizontal=False,
        key="real_image",
        label_visibility="hidden",
    )
    st.text_input(
        "In what way is the image not genuine?",
        key="real_image_text",
        placeholder="e.g., edited, AI generated, etc.",
        disabled=st.session_state.real_image != QUESTION_OPTIONS["real_image"][1],
    )

    st.divider()
    st.write(
        "**Does the image originate from a reliable and verified source? A non-reliable source may include imposters or satirical platforms.**"
    )
    st.radio(
        "genuine source",
        QUESTION_OPTIONS["real_source"],
        index=0,
        horizontal=False,
        key="real_source",
        label_visibility="hidden",
    )
    st.text_input(
        "In what way is the source not trustworthy?",
        key="real_source_text",
        placeholder="e.g., imposter, satire, etc.",
        disabled=st.session_state.real_source != QUESTION_OPTIONS["real_source"][1],
    )
with image_text_focused:
    st.write(
        "**Does the claim in the *tweet's text* faithfully represent the content of the image?**"
    )
    st.radio(
        "misleading tweet text",
        QUESTION_OPTIONS["tweet_text"],
        index=0,
        horizontal=False,
        key="tweet_text",
        label_visibility="hidden",
    )
    st.text_input(
        "In what way does the tweet not faithfully represent the content of the image?",
        key="tweet_text_text",
        placeholder="e.g., time/place mismatch, misleading interpretation, etc.",
        disabled=st.session_state.tweet_text != QUESTION_OPTIONS["tweet_text"][1],
    )
    st.divider()
    st.write(
        "***If the image contains a textual claim*, does the claim faithfully represent the visual content of the image?**"
    )
    st.radio(
        "misleading image text",
        QUESTION_OPTIONS["embedded_text"],
        index=0,
        horizontal=False,
        key="embedded_text",
        label_visibility="hidden",
    )
    st.text_input(
        "In what way does the text not faithfully represent the visual content of the image?",
        key="embedded_text_text",
        placeholder="e.g., time/place mismatch, misleading interpretation, etc.",
        disabled=st.session_state.embedded_text != QUESTION_OPTIONS["embedded_text"][2],
    )


st.divider()
st.checkbox("It is impossible to annotate this image", key="cannot_annotate")
st.text_input(
    "Explain why:",
    key="cannot_annotate_text",
    placeholder="Optional",
    disabled=not st.session_state.cannot_annotate,
)


st.button(
    "**Confirm**",
    on_click=lambda: confirm_label(note=note),
    key="confirm_button",
    use_container_width=True,
    type="primary",
)
