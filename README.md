\# Real-Time Detailed Video Analysis of Fruit Flies



\## Project Overview



This project implements a machine learning and computer vision pipeline for analyzing fruit-fly videos.



The system processes video frames and attempts to determine:



\- Number of flies present

\- Sex of detected flies

\- Fly orientation

\- Male wing angle



The project is based on the Stanford CS229 fruit-fly video analysis project and implements a simplified version of the original approach using OpenCV and machine learning.



\---



\## Problem Statement



Detailed analysis of fruit-fly behavior from video can require manual observation and annotation. The objective of this project is to develop an automated pipeline that can process fruit-fly video frames and extract useful information about the flies.



The implemented system combines image preprocessing, contour detection, feature extraction and machine learning models to analyze the video.



\---



\## Objectives



The main objectives are:



1\. Detect fruit flies from video frames.

2\. Estimate the number of flies in each frame.

3\. Classify detected flies as male or female.

4\. Estimate fly orientation.

5\. Estimate male wing angle.

6\. Process video frames in real time.

7\. Save an annotated output video showing the analysis.



\---



\## Dataset



The project uses the fruit-fly dataset associated with the Stanford CS229 project.



The dataset contains:



\- Grayscale fruit-fly images

\- Video files

\- JSON annotation files

\- Manually annotated fly landmarks



The processed dataset contains:



\- 326 JSON annotation files

\- 326 corresponding images

\- 2412 annotated points



The annotations contain landmarks such as:



\- Male head (`mh`)

\- Male body point (`mp`)

\- Male abdomen (`ma`)

\- Female body point (`fp`)

\- Male wing points (`mw`)

\- Additional male body point (`mp2`)



The dataset is not included in this GitHub repository because of its size.



\---



\## Technologies Used



\- Python

\- OpenCV

\- NumPy

\- Scikit-learn

\- Scikit-image

\- Pandas

\- Joblib

\- Matplotlib



\---



\## Project Structure



```text

fruit-fly-video-analysis/

│

├── data/

│   └── sample/

│

├── models/

│   ├── fly\_count\_model.pkl

│   ├── sex\_model.pkl

│   ├── orientation\_model.pkl

│   └── wing\_angle\_model.pkl

│

├── notebooks/

│

├── results/

│   ├── metrics/

│   ├── plots/

│   └── videos/

│

├── src/

│   ├── preprocessing.py

│   ├── feature\_engineering.py

│   ├── fly\_count.py

│   ├── sex\_classifier.py

│   ├── orientation.py

│   ├── wing\_angle.py

│   └── pipeline.py

│

├── train/

│   ├── train\_fly\_count.py

│   ├── train\_sex.py

│   ├── train\_orientation.py

│   └── train\_wing\_angle.py

│

├── README.md

├── requirements.txt

├── .gitignore

└── main.py

