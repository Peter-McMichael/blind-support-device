<div align="center">
  <h1>Blind Vision Support Device Using ML and AI</h1>
</div>

<img width="800" height="515" alt="image" src="https://github.com/user-attachments/assets/48cf0357-ab7b-415a-8c41-90be93a1bd2f" />



<p>This project uses a webcam and trained dataset to recognize and identify objects and alert the user about what's in front of them. Made with NVIDIA Jetson Orin Nano. Could potentially be integrated into glasses, measure depth, include multiple languages, have more commands, and be used to guide visually impaired people. Code is entirely Python. Webcam is Logitech C270 HD Webcam. </p><br>
<br>
<p> Uses pyttsx3 library to read the detected item to the user. Uses Speech Recognition library to listen to user commands. </p> <br>
<p>- Uses Pascal VOC 2012 Dataset for models</p> <br>
<p> Link to Pascal VOC 2012 Dataset on Kaggle: https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset</p>
<br>

<p>- Dataset trained and exported from .pth to .onnx file</p>


<br>
<br>
<br>
thr
<h1> How does it work? </h1>
<p> When an item is recognized, it is sent to the python script and read aloud with pyttsx3. If the spacebar is clicked, a voice recorder is turned on through speech recognition. If the a commmand is recognized (currently the two commands are "Volume up" and "Volume down"), a certain action like an increase or decrease in volume for the audio occurs. There is a cooldown for the distance between re-announcing the item in front of you. If no camera is found, an error message if played. Only the object with the highest confidence is announced. Every item that is announed in also logged in the terminal. </p>


[View a video explanation here](video link)
