import torch
# Load Florence-2
from transformers import AutoModelForCausalLM, AutoProcessor

from utils import plot_bbox, get_random_frame

# Load Model
def load_detection_model():
    model_id = 'microsoft/Florence-2-large'
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model,processor

# Execute using given prompts
def run(task_prompt, text_input=None, image = None, model = None, processor = None):
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

def caption_grounding(video_path, model, processor, captions = None):
    image = get_random_frame(video_path)
    text_input = captions
    task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
    results = run(task_prompt=task_prompt, text_input=text_input, image= image, model= model, processor= processor)
    results['<DETAILED_CAPTION>'] = text_input
    plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])
    return results['<CAPTION_TO_PHRASE_GROUNDING>']['labels']
