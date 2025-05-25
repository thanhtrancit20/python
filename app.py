from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import face_recognition
import numpy as np

app = Flask(__name__)
CORS(app)

# Folder to store registered face images and encodings
students_images_dir = 'students_images'
os.makedirs(students_images_dir, exist_ok=True)

@app.route('/register-face', methods=['POST'])
def register_face():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    student_id = request.form.get('student_id')

    if not student_id:
        return jsonify({'error': 'Student ID is required'}), 400

    # Load and encode image
    user_image = face_recognition.load_image_file(image_file)
    user_face_encodings = face_recognition.face_encodings(user_image)

    if len(user_face_encodings) > 0:
        user_face_encoding = user_face_encodings[0]

        # Save image
        image_path = os.path.join(students_images_dir, f"{student_id}.jpg")
        image_file.seek(0)  # Reset stream position
        image_file.save(image_path)

        # Save face encoding
        encoding_path = os.path.join(students_images_dir, f"{student_id}_encoding.npy")
        np.save(encoding_path, user_face_encoding)

        return jsonify({'message': 'Face registered successfully', 'student_id': student_id})
    else:
        return jsonify({'error': 'No faces found in the image'}), 400


@app.route('/login-face', methods=['POST'])
def login_face():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    student_id = request.form.get('student_id')
    if not student_id:
        return jsonify({'error': 'No student ID provided'}), 400

    image_file = request.files['image']
    unknown_image = face_recognition.load_image_file(image_file)
    unknown_face_encodings = face_recognition.face_encodings(unknown_image)

    if len(unknown_face_encodings) == 0:
        return jsonify({'error': 'No faces found in the image'}), 400

    unknown_face_encoding = unknown_face_encodings[0]

    encoding_path = os.path.join(students_images_dir, f"{student_id}_encoding.npy")
    if not os.path.exists(encoding_path):
        return jsonify({'error': 'Student encoding not found'}), 404

    stored_encoding = np.load(encoding_path)
    results = face_recognition.compare_faces([stored_encoding], unknown_face_encoding)

    if results[0]:
        return jsonify({'success': True, 'student_id': student_id}), 200

    return jsonify({'success': False, 'error': 'Face not recognized'}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
