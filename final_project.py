import jetson_inference
import jetson_utils
import pyttsx3
import time
# import sounddevice as sd
import keyboard
import speech_recognition as sr


# Set up detection network and video sources
net = jetson_inference.detectNet(argv=[
    "--model=/home/nvidia4/jetson-inference/python/training/detection/ssd/models/test_detect_model/ssd-mobilenet.onnx",
    "--labels=/home/nvidia4/jetson-inference/python/training/detection/ssd/models/test_detect_model/labels.txt",
    "--input-blob=input_0",
    "--output-cvg=scores",
    "--output-bbox=boxes"
], threshold=0.5)
camera = jetson_utils.videoSource("/dev/video1")
print(camera.IsStreaming())
display = jetson_utils.videoOutput("output.mp4", argv=["--headless"])

# Set up TTS engine
tts = pyttsx3.init()
recognizer = sr.Recognizer()
mic = sr.Microphone()
last_spoken = None
last_time = 0
# engine = pyttsx3.init()
cooldown = 3 
volume = 0.7

tts.say("Welcome! Click the spacebar to activate a command! You can say VOLUME UP and VOLUME DOWN.")
tts.runAndWait()

while True:
    video = camera.Capture()


    if keyboard.is_pressed('space'):
        print("Microphone activated.")
        with mic as source:
            try:
                audio = recognizer.listen(source, timeout = 5)
            except sr.WaitTimeoutError:
                audio = None
        if audio == None:
            command=""
        else:
            try:
                command = recognizer.recognize_google(audio).lower()
            except (sr.UnknownValueError, sr.RequestError):
                command = ""

            #here

        if command in ("volume up", "volume increase", "increase volume"):
            if volume < 1:
                volume += 0.3
            else:
                tts.say("Maximum volume.")
                tts.runAndWait()
        elif command in ("volume down", "volume decrease", "decrease volume"):
            if volume > 0.1: # 1 so that the volume cant be completely turned off.
                volume -= 0.3
            else:
                tts.say("Minimum volume.")
                tts.runAndWait()
        tts.setProperty('volume', volume)

    if video is None:
        print("Error: there is no camera!")
        tts.say("No camera found.")
        tts.runAndWait()
        continue
    detections = net.Detect(video) #IDs the object in front of the camera.
    display.Render(video) #live display
    if not camera.IsStreaming() or not display.IsStreaming():
        break


    if detections:
        top = max(detections, key=lambda d: d.Confidence)
        label = net.GetClassDesc(top.ClassID)


        #timer
        now = time.time()
        if label != last_spoken or (now-last_time) > cooldown: # if a new object is identified or the cooldown timer has been reached.
            print(f"Detected: {label} ({top.Confidence:.2f})") 
            tts.setProperty('volume', volume)
            tts.say(label) # print AND say label because blind person clearly can't read
            tts.runAndWait()

            last_spoken = label # update last_spoken
            last_time = now #update last_time