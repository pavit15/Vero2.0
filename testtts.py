import pyttsx3

engine = pyttsx3.init()

# Set slower rate and volume
engine.setProperty('rate', 130)
engine.setProperty('volume', 0.9)

# Loop to find and set "English (Great Britain)" voice
voices = engine.getProperty('voices')
for voice in voices:
    if "Great Britain" in voice.name:
        engine.setProperty('voice', voice.id)
        break

engine.say("Hello! This is the English Great Britain voice.")
engine.runAndWait()
