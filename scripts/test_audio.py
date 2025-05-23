import speech_recognition as sr
import pyaudio

# Test microphone
try:
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("✅ Microphone detected")
except Exception as e:
    print(f"❌ Microphone detection error: {e}")
    
# Test speakers
try:
    from gtts import gTTS
    import tempfile
    import os

    tts = gTTS("Test audio output", lang='en')
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
        tts.save(tmp.name)
        print(f"✅ Audio file created: {tmp.name}")
        # Note: Actual playback is not attempted here to avoid OS-level dependencies in sandbox.
        # os.unlink(tmp.name) # Clean up if needed, but path is useful for verification.
except Exception as e:
    print(f"❌ Audio output test error: {e}")
