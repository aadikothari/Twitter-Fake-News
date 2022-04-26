# Twitter-Fake-News

We have provided a requirements.txt file using the pipreqs function. The ego machine was running Windows 10 and the commands were run by Powershell.
The same was tested in a macOS and WSL (Windows Subsystem for Linux) based environment.

Our version of code was ran on Python 3.10.0 and pip 22.0.4

STEPS to RUN:

1) CLONE the repo/DOWNLOAD the files (Make sure to be in the Twitter-Fake-News folder)
2) RUN the command 'pip install -r requirements.txt' // 'pip3 install -r requirements.txt'
3) OBSERVE that all libraries install successfully.
4) RUN the command 'python final.py' // 'python3 final.py' or run the file using an IDE interpreter
5) At the very bottom, the confusion matrix for the Decision Tree model will be commented out, but feel free to uncomment to check for accuracy of the model. This can be moved a notch above to measure the confusion matrix for the Neural Network model

While running the file, NOTICE 1) Comments. We kept helper methods for some functions from towardsdatascience.com and geeksforgeeks so as to aid us if needed.
However, those helper functions (if commented) are not used. The archive/ folder contains the dataset for the Fake/Real news that our model is trained on.

Comments explaining the code flowthrough are included. Our script runs Neural Network and Decision Tree classifers. Libraries used are specified in the requirements.txt file.
