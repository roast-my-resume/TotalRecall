import os
import spacy
import av
import cv2
import random
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
from matplotlib import pyplot as plt, patches as patches

def extract_noun(labels):
    """
    Extract nouns based on given labels using spacy
    """
    nlp = spacy.load("en_core_web_sm")
    single_noun_labels = []
    for label in labels:
        doc = nlp(label)
        nouns = [token for token in doc if token.pos_ == "NOUN"]
        if nouns:
            core_noun = min(nouns, key=lambda token: token == token.head)
            single_noun_labels.append(core_noun.text)

    # print(single_noun_labels)
    return single_noun_labels

def generate_emoji(prompt):
    """
    Generate emoji based on given prompt
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    client = OpenAI(
        api_key=api_key
    )
    # construct prompt
    gpt_prompt = f"""
    Given a short text, return the most suitable emoji that represents the text. 
    For example:
    - "happy birthday" -> 🎂

    Now respond to the following input:
    - "{prompt}" ->
    """

    try:
        # use GPT-4 API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an assistant trained to generate emojis for short text."
                                              "Focus on the objects and events described in the text."},
                {"role": "user", "content": gpt_prompt},
            ],
            max_tokens=10,
            temperature=0,
        )
        # extract emoji
        emoji = response.choices[0].message.content[0]
        return emoji
    except Exception as e:
        print(f"Error: {e}")
        return None

# conversation template for content
conversation1 = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Describe the video content strictly using one of the following templates:\n"
                    "Content: The video shows {} {} while {}.\n"
                    "Content: A {} can be seen {} {}.\n"
                    "Content: This clip features {} {} in {}.\n"
                    "Content: The scene captures {} {} during {}.\n"
                    "Please ensure your response adheres to one of these templates."
                ),
            },
            {"type": "video"},
            ],
    }
]
# conversation template for title and type
conversation2 = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Analyze the video and generate a short title summarizing its main content and an event type"
                    "categorizing the content. Please format your response as follows:\n\n"
                    "Title: <short_title>\n"
                    "Event Type: <event_type>\n"
                    "Examples:\n"
                    "  Title: Children playing soccer match\n"
                    "  Event Type: Sports\n"
                    "Provide your response strictly in this format."
                ),
            },
            {"type": "video"},
        ],
    },
]


def validate_video_path(video_path: str):
    """
    Validate the video path to ensure it exists and is a valid video file.

    Args:
        video_path (str): Path to the video file.

    Returns:
        bool: True if the path is valid, False otherwise.
    """
    # check for existence
    if not os.path.exists(video_path):
        print(f"Error: The file at path '{video_path}' does not exist.")
        return False

    # check extension
    valid_extensions = [".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv"]
    if not any(video_path.lower().endswith(ext) for ext in valid_extensions):
        print(f"Error: The file '{video_path}' is not a recognized video format.")
        return False

    # try pyav
    try:
        container = av.open(video_path)
        container.close()  # close after successfully loaded
    except av.AVError as e:
        print(f"Error: The file at path '{video_path}' could not be opened as a video. Details: {e}")
        return False

    # pass
    return True


def plot_bbox(image, data):
    # Create a figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Plot each bounding box
    for bbox, label in zip(data['bboxes'], data['labels']):
        # Unpack the bounding box coordinates
        x1, y1, x2, y2 = bbox
        # Create a Rectangle patch
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        # Add the rectangle to the Axes
        ax.add_patch(rect)
        # Annotate the label
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

        # Remove the axis ticks and labels
    ax.axis('off')

    # Show the plot
    plt.show()


def get_random_frame(video_path):
    # open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        raise ValueError("No frame in the video")

    # random select a frame
    random_frame_number = random.randint(0, frame_count - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)

    # read random frame
    success, frame = cap.read()
    if not success:
        print(f"Unable to read frame: {random_frame_number}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, frame = cap.read()

    cap.release()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # convert to PIL
    image = Image.fromarray(frame)

    return image
