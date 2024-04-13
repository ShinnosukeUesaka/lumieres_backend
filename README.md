# lumieres_backend

## DEMO

https://github.com/ShinnosukeUesaka/lumieres_backend/assets/45286939/5190e232-76ab-4c5b-8156-65624260790f

## Inspiration
All of us watched tutorials on YouTube to learn concepts otherwise difficult to understand. But science tells us that watching a video is not enough. How many times have we watched a tutorial convinced that we had understood everything...and then the day after it is all gone. Don't worry, we have a solution for you. Interactive learning (doing not only watching) is the best way, according to recent neuroscientific research, to remember longer and understand deeper the topics studied.

## What it does
Introducing Luminieres, the first assistant built to improve educational videos. The content of many educational videos is top-tier, but only if mixed with practice we will remember and understand at best. With Luminieres, at certain timestamps, the video will be interrupted and an avatar of the speaker will ask you questions regarding what was being explained at that point. Questions can either be with options or free answers. To every answer, correct or incorrect it may be, the user will receive feedback from the speaker.

## How we built it
Frontend in Next.js
The video gets transcripted by Whisper AI.
The transcript is fed to the Large Mistral model through API. The model uses a custom prompt that explains in detail how to build a certain number of questions and the way it should give feedback
The response is fetched and it is converted into audio format through Eleven Labs.
This AI-generated audio and the video itself is used by sadtalker API to construct the avatar of the speaker.
What's next for Lumiere
Make it more general: not only YouTube videos
Make it into a Chrome Extension

## Frontend
https://github.com/tsengtinghan/lumieres



