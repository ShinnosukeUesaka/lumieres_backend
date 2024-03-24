import dotenv
dotenv.load_dotenv('lumineres/.env')

from pytube import YouTube
from openai import OpenAI
from pydub import AudioSegment
import pickle
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import os
import json
import fastapi
from elevenlabs.client import ElevenLabs
import elevenlabs
import cv2
import replicate
import uuid
from concurrent.futures import ThreadPoolExecutor
import time
import traceback
import pydantic 

app = fastapi.FastAPI()

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"
client = OpenAI()


class Item(pydantic.BaseModel):
    url: str
    max_questions: int = 2

@app.post("/create_questions")
async def create_questions(input: Item):
    return {'questions': process_url(input.url, input.max_questions)}

def find_best_matching_section(sentence, transcript):
    sentence_words = sentence.split()
    best_match_end_timestamp = None
    best_match_count = 0

    for i in range(len(transcript) - len(sentence_words) + 1):
        transcript_snippet = transcript[i:i + len(sentence_words)]
        match_count = 0

        for entry in transcript_snippet:
            if entry['word'].lower() in [word.lower() for word in sentence_words]:
                match_count += 1

        if match_count > best_match_count:
            best_match_count = match_count
            best_match_end_timestamp = transcript_snippet[-1]['end']

    return best_match_end_timestamp



EXAMPLE_JSON = """{
    "questions": [
        {
            "question": "What is the best French cheese?",
            "choices": [
                "Brie",
                "Camembert",
                "Roquefort",
                "Mimolette"
            ],
            "answer": "Roquefort",
            "question_script": "What is the best French cheese?",
            "correct_answer_script": "You got it! The best French cheese is Roquefort.",
            "incorrect_answer_script": "Actually, the best French cheese is Roquefort. Roquefort is a sheep's milk cheese from the south of France. It has a tangy, salty flavor and a creamy texture. It's often crumbled over salads or melted into sauces. Try it next time you're at the cheese counter!",
            "sentence_in_transcription_before_asking": "That concludes the section on French cheese. Now let's move on to the next topic."
        }
    ]
}"""

def clone_audio(audio_path) -> str:
    print("cloning voice")
    audio_client = ElevenLabs()

    voice = audio_client.clone(
        name=uuid.uuid4().hex,
        description="An american male voice",
        files=[audio_path],
    )

    return voice



def get_face_image(url, output_path="face.png"):
    print("getting face image")
    YouTube(url).streams.get_highest_resolution().download(filename='video.mp4')

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
        cv2.imwrite(output_path, face_image)
    else:
        print("No face detected in the first frame.")
        exit()

    video.release()
    return output_path

def save_audio_get_transcription(url: str = "https://www.youtube.com/watch?v=zjkBMFhNj_g", audio_file_path="audio.mp3"):
    print("creating transcription")
    # downalod audio and save it to audio.mp3
    YouTube(url).streams.filter(only_audio=True).first().download(filename='audio.mp3')
    # cut the audio to 10 minutes
    audio = AudioSegment.from_file("audio.mp3")
    audio = audio[:10*60*1000]
    audio.export(audio_file_path, format="mp3")

    audio_file = open("audio.mp3", "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file,
        response_format="verbose_json",
        timestamp_granularities=["word"]
    )
    return transcription

def create_questions(transcription):
    print("generating questions")
  
    text = transcription.text
    
    client = MistralClient(api_key=api_key)
    

    PROMPT = f"""This is a transcription of an video. Your job is to generate *two* multiple choice questoins and openended questions based on the transcription.
The question will be shown to the user as they watch the video. The question should be displayed to the user at the end of each section of the video.
The script will be read aloud, try to mimic the tone and style of the script. The answer to the question should be one of the multiple choice options.
Example of your json response
{EXAMPLE_JSON}

Transcription"
{text}
"""

    messages = [
        ChatMessage(role="user", content=PROMPT)
    ]

    chat_response = client.chat(
        model=model,
        response_format={"type": "json_object"},
        messages=messages,
    )

    print(chat_response.choices[0].message.content)
    # convert string to json
    try:
        json_response = json.loads(chat_response.choices[0].message.content)
    except:
        print("Error converting string to json")
        print(chat_response.choices[0].message.content)
        # try again
        chat_response = client.chat(
            model=model,
            response_format={"type": "json_object"},
            messages=messages,
        )
        try:
            json_response = json.loads(chat_response.choices[0].message.content)
        except:
            raise Exception("Error converting string to json")
    
    # save json as a file
    with open("questions.json", "w") as f:
        json.dump(json_response, f)
    
    # load json from file
    with open("transcription.pkl", "rb") as f:
        transcription = pickle.load(f)
    
    with open("questions.json", "r") as f:
        json_response = json.load(f)
        
    final_questions = []
    for question in json_response["questions"]:
        timestamp = find_best_matching_section(question["sentence_in_transcription_before_asking"], transcription.words)
        final_questions.append({
            "question": {
                "type": "multiple_choice",
                "question": question["question"],
                "options": question["choices"],
                "answer": question["answer"],
                "question_script": question["question_script"],
                "correct_answer_script": question["correct_answer_script"],
                "incorrect_answer_script": question["incorrect_answer_script"]
            },
            "timestamp": timestamp
        })
    
    return final_questions


    
    
def create_video_from_text(face_image_path: str, text: str, voice_id: str):
    audio_client = ElevenLabs()
    audio = audio_client.generate(text=text, voice=voice_id)
    audio_file_name =  f"{uuid.uuid4().hex}.mp3"
    elevenlabs.save(audio, audio_file_name)
    image = open(face_image_path, "rb")
    audio = open(audio_file_name, "rb")
    input = {
        "driven_audio": audio,
        "source_image": image,
        "enhancer": "RestoreFormer"
    }
    
    output = replicate.run(
        "lucataco/sadtalker:85c698db7c0a66d5011435d0191db323034e1da04b912a6d365833141b6a285b",
        input=input
    )
    return output


def create_videos(face_image_path, script, voice_id):
    return create_video_from_text(face_image_path, script, voice_id)

def create_videos_for_question(index, question, face_image_path, voice_id):
    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            'question_video': executor.submit(create_videos, face_image_path, question["question"]["question_script"], voice_id),
            'correct_answer_video': executor.submit(create_videos, face_image_path, question["question"]["correct_answer_script"], voice_id),
            'wrong_answer_video': executor.submit(create_videos, face_image_path, question["question"]["incorrect_answer_script"], voice_id)
        }

        for key, future in futures.items():
            try:
                results[key] = future.result()  # This blocks until the future is done
            except Exception as e:
                print(f"An error occurred in {key} for question index {index}: {e}")
                traceback.print_exc()  # This prints the traceback of the error

    return (index, {
        "question_video_url": results.get('question_video', 'Error'),
        "correct_answer_video_url": results.get('correct_answer_video', 'Error'),
        "wrong_answer_video_url": results.get('wrong_answer_video', 'Error')
    })


def process_url(url: str = "https://www.youtube.com/watch?v=zjkBMFhNj_g", max_questions=2):
    face_image_path = "face.png"
    audio_file_path = "audio.mp3"
    transcription = save_audio_get_transcription(url, audio_file_path)
    voice_id = clone_audio(audio_file_path)
    print(f"voice_id: {voice_id}")
    questions = create_questions(transcription)
    print(questions)
    
    # limit the number of questions 
    questions = questions[:max_questions]
    
    get_face_image(url, output_path="face.png")
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Launch parallel tasks for each question
        futures = [executor.submit(create_videos_for_question, index, question, face_image_path, voice_id) for index, question in enumerate(questions)]

        # Wait for all tasks to complete and update the questions list
        for future in futures:
            index, video_urls = future.result()
            questions[index]["question"].update(video_urls)

    return questions
        

if __name__ == "__main__":
    print(process_url())
    