import cv2
import pafy
import vlc
import time
import os

# Set PAFY_BACKEND to "internal"
os.environ['PAFY_BACKEND'] = 'internal'

# Function to detect face using OpenCV
def detect_face():
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Capture video from the webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Return True if a face is detected, otherwise return False
        if len(faces) > 0:
            cap.release()  # Release the capture
            return True
        else:
            cap.release()  # Release the capture
            return False

    # Release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Function to play YouTube video using Pafy and VLC
def play_youtube_video(url):
    # Create a video object using Pafy
    video = pafy.new(url)
    
    # Get the best available stream
    best = video.getbest()
    
    # Create a VLC media player object
    media = vlc.MediaPlayer(best.url)
    
    # Play the video
    media.play()
    
    return media

# Main function to control video playback based on face detection
def main():
    video_url = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'  # Example YouTube URL
    media = play_youtube_video(video_url)
    
    while True:
        face_detected = detect_face()
        
        if face_detected:
            print("Face detected, playing video.")
            media.play()
        else:
            print("No face detected, pausing video.")
            media.pause()
        
        time.sleep(1)  # Adjust the sleep time as necessary

if __name__ == "__main__":
    main()
