import os
from flask_cors import CORS
from flask import Flask, request, jsonify
from ObjectDetection import load_detection_model, caption_grounding
from VideoInference import process_video, load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)
inference_model, inference_processor = load_model()
detection_model, detection_processor = load_detection_model()
# pass temp video clip to inference model
def video_to_text(video_path: str):
    results = process_video(video_path,inference_model,inference_processor)
    results["objects"] = caption_grounding(video_path = video_path, model = detection_model, processor = detection_processor, captions = results["description"])
    print(results)
    return {
        "emoji": results["emoji"],
        "type": results["type"],
        "title": results["title"],
        "description": results["description"],
        "objects": results["objects"]
    }

@app.route('/video-to-text', methods=['POST'])
def analyze_video():
    # Save uploaded file with secure filename
    video_file = request.files['video']
    video_path = os.path.join(temp_dir, secure_filename(video_file.filename))
    video_file.save(video_path)
    
    # Generate response 
    response = video_to_text(video_path)

    # Delete temporary file
    # os.remove(video_path)     # fixme: uncomment this line to delete the file after processing
    
    # Return response as JSON
    return jsonify(response)
    

if __name__ == '__main__':
    temp_dir = "./temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    app.run(host='localhost', port=10708, debug=True)

