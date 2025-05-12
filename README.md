Emotion-Aware Speech Recognition and Face Emotion Detection
This project implements emotion-aware speech recognition using the Whisper ASR (Automatic Speech Recognition) model and emotion detection from faces using a pretrained model. It integrates both audio and visual emotion recognition systems to provide a more comprehensive understanding of emotions in multimodal content (audio and visual).

Features:
Speech Transcription: Uses the Whisper ASR model to transcribe speech from audio files.

Emotion Detection from Audio: Extracts audio features (MFCC, pitch) and classifies emotions (happy, sad, angry, neutral) from the speech using a Support Vector Classifier (SVC).

Face Emotion Detection: Detects faces from images and classifies emotions using a pretrained emotion model (Mini-XCEPTION).

Multimodal Emotion Awareness: Provides emotion recognition from both speech and facial expressions.

Prerequisites
To run this project, you will need to install the following Python libraries:

pip install -q openai-whisper moviepy librosa scikit-learn keras opencv-python
pip install whisper

Setup
Install dependencies:

Make sure to install all the necessary libraries before running the program. You can install them using the pip commands mentioned above.

Pretrained Models:

The code uses a pretrained Whisper model for speech-to-text conversion.

For face emotion detection, the project uses a pretrained Mini-XCEPTION model.

Running the Code:

The code works in a Jupyter Notebook or Google Colab environment.

Upload an MP4 file for speech transcription and audio emotion detection.

Upload a face image (JPG/PNG) for visual emotion detection.

Steps
Transcribe Speech:

The program transcribes speech from an audio file (MP4 format) using Whisper ASR.

Classify Emotion from Audio:

It extracts audio features like MFCC and pitch from the transcribed speech and classifies the emotion of the speaker.

Detect Emotion from Faces:

The program uses OpenCV to detect faces from images and uses a pretrained Mini-XCEPTION model to classify the emotion of the person in the image.

Multimodal Emotion Awareness:

It provides the detected emotion from both the audio (speech) and the visual (face) input.

Files
emotion_model.h5: Pretrained face emotion detection model.

emotion_recognition.py: The script containing all the functions and logic for emotion detection.

Example Workflow
Upload MP4 file:

The user uploads an MP4 file (speech or video file), and the program will transcribe the speech and classify the emotion from the audio.

Upload Face Image:

The user uploads a face image (JPG/PNG), and the program will detect the face(s) and classify the emotion shown.

Code Explanation
Transcribe Audio: The function transcribe_audio() uses the Whisper ASR model to transcribe speech from the audio file.

Feature Extraction: The function extract_audio_features() extracts MFCC and pitch features from the audio file for emotion classification.

Emotion Classification: The emotion classifier is a Support Vector Classifier (SVC) trained on synthetic data, used in classify_emotion() to detect emotions like happy, sad, angry, or neutral.

Face Emotion Detection: Using OpenCV and a pretrained Mini-XCEPTION model, the face emotion detection is handled by detect_face_emotion().

Example Outputs

Speech Emotion Detection:

Transcription: Hello! I am so happy today.
Detected Emotion (Audio): happy
Detected Language: en
Face Emotion Detection:

Detected Emotion (Face): Happy
License
This project is open source and available under the MIT License.
