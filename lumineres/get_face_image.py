import cv2
from pytube import YouTube
import replicate
import requests
from pydub import AudioSegment
from elevenlabs.client import ElevenLabs




def create_video_from_texts(url, texts: list[str]):
    
    YouTube(url).streams.filter(only_audio=True).first().download(filename='audio.mp3')
    # cut the audio to 10 minutes
    audio = AudioSegment.from_file("audio.mp3")
    audio = audio[:10*60*1000]
    audio.export("audio.mp3", format="mp3")
    # clone voice

    # generate video and audio in paralell but do it in parallel
     


if __name__ == '__main__':
    url = "https://www.youtube.com/watch?v=zjkBMFhNj_g"
    #url ="https://www.youtube.com/watch?v=NJeFmj2ZjZs"
    #YouTube(url).streams.get_highest_resolution().download(filename='video.mp4')

    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open the video file
    video = cv2.VideoCapture('video.mp4')

    # Get the video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Calculate the number of frames in the first minute
    frames_in_minute = fps * 60

    # Create a VideoWriter object for saving the cropped video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #output_video = cv2.VideoWriter('cropped_video.mp4', fourcc, fps, (width, height))

    # # Read the first frame from the video
    # ret, first_frame = video.read()
    # read the 20 second frame
    video.set(cv2.CAP_PROP_POS_MSEC, 20000)
    ret, first_frame = video.read()

    # Convert the first frame to grayscale
    gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Get the coordinates of the first detected face
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        resize_ratio = 1.5
        # make the region bigger by resize_ratio
        x = int(x - (resize_ratio - 1) * w/2)
        y = int(y - (resize_ratio - 1) * h/2)
        w = int(w * resize_ratio)
        h = int(h * resize_ratio)
        # Extract the face region from the first frame
        face_image = first_frame[y:y+h, x:x+w]
        # Save the face image as a PNG file
        cv2.imwrite('face.png', face_image)
    else:
        print("No face detected in the first frame.")
        exit()

    video.release()
    
    image = open("face.png", "rb")
    input = {
        "driven_audio": "https://replicate.delivery/pbxt/Jf1gczNATWiC94VPrsTTLuXI0ZmtuZ6k0aWBcQpr7VuRc5f3/japanese.wav",
        "source_image": image,
        "enhancer": "RestoreFormer"
    }
    
    output = replicate.run(
        "lucataco/sadtalker:85c698db7c0a66d5011435d0191db323034e1da04b912a6d365833141b6a285b",
        input=input
    )
    print(output)
    # output is url of the video in mp4, download the video
    video = requests.get(output)
    open("output.mp4", "wb").write(video.content)
