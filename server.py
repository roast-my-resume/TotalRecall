from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time

from torch import inference_mode

from ObjectDetection import processor
from VideoInference import process_video, load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# pass temp video clip to inference model
def video_to_text(video_path: str):
    inference_model, inference_processor = load_model()
    results = process_video(video_path,inference_model,inference_processor)
    return {
        "type": results["type"],
        "title": results["title"],
        "description": results["description"]
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

