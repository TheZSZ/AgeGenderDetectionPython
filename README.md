# AgeGenderDetectionPython
An age and gender detection program in Python that I created to learn about the OpenCV Library and Deep Learning

Required Libraries:
- OpenCV (needs to be installed)
- OS (included with python)
- Sys (included with python) 

Library Installation with pip package manager:
- 'pip install opencv-python' to install OpenCV

Commands:
- 'python ageDetect.py' to see usage cases
- 'python ageDetect.py [path/to/file]' to pass an image into the program
- 'python ageDetect.py camera' to pass camera stream into the program



Comments:
- *Mostly* works correctly
![Screenshot](images/billieeilish.jpg)
![Screenshot](images/billieeilish-detection.png)
![Screenshot](images/grandson.jpg)
![Screenshot](images/grandson-detection.png)
![Screenshot](images/kanye.jpg)
![Screenshot](images/kanye-detection.png)
![Screenshot](images/logic.jpg)
![Screenshot](images/logic-detection.png)
![Screenshot](images/macklemore.jpg)
![Screenshot](images/macklemore-detection.png)
![Screenshot](images/richbrian.jpg)
![Screenshot](images/richbrian-detection.png)

- Semi-realistic cartoon charater (Heavy from Team Fortress 2)
![Screenshot](images/tf2heavy.jpg)
![Screenshot](images/tf2heavy-detection.png)
- Works with multiple people
![Screenshot](images/HouseoftheRisingSun.jpg)
![Screenshot](images/HouseoftheRisingSun-detection.png)
![Screenshot](images/SabreFencers.jpg)
![Screenshot](images/SabreFencers-detection.png)
- Cannot detect faces if wearing objects around or on face like Hijab or if side profile :(
