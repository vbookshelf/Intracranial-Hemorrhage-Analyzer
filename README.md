## Intracranial-Hemorrhage-Analyzer
Ai powered web app that detects and segments intracranial hemorrhages on brain CT images - Tensorflow.js

Live Web App: http://brain.test.woza.work/

<i>For best results please use the Chrome browser.<br>
In other browsers the app may freeze.</i>



<br>

<img src="http://brain.test.woza.work/assets/githubimage.png" width="500"></img>

<br>

My goal for this project was to build a prototype tensorflow.js web app and deploy it online. The app automatically detects and segments intracranial hemorrhages in brain CT scans. It takes as input a single jpg or png image (brain window) and outputs a segmentation showing the area where bleeding has been detected.

The process used to build and train the model is described in this Kaggle kernel:<br>
https://www.kaggle.com/vbookshelf/intracranial-hemorrhage-analyzer-tfjs-web-app

The dataset used to train the model can be found on Kaggle:<br>
https://www.kaggle.com/vbookshelf/computed-tomography-ct-images



All javascript, html and css files used to create the web app are available in this repo.
