dirs = "frames/"
# Load Florence-2
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)



def run(task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=inputs["pixel_values"].cuda(),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer

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

def caption_grounding(image):
    task_prompt = '<CAPTION>' # base
    # task_prompt = '<DETAILED_CAPTION>' # more specific
    # task_prompt = '<MORE_DETAILED_CAPTION>' # most specific
    results = run(task_prompt)
    text_input = results[task_prompt]
    task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
    results = run(task_prompt, text_input)
    results['<DETAILED_CAPTION>'] = text_input
    plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])

if __name__ == '__main__':
    image = Image.open(dirs + '00000001.jpg')
    # results = run(task_prompt)
    caption_grounding(image = image)
