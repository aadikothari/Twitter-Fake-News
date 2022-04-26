# Twitter-Fake-News

We have provided a requirements.txt file using the pipreqs function. The ego machine was running Windows 10 and the commands were run by Powershell.
The same was tested in a macOS and WSL (Windows Subsystem for Linux) based environment.

Our version of code was ran on Python 3.10.0 and pip 22.0.4

STEPS to RUN:

1) CLONE the repo/DOWNLOAD the files (Make sure to be in the Twitter-Fake-News folder)
2) RUN the command 'pip install -r requirements.txt'
3) OBSERVE that all libraries install
4) RUN the command 'python final.py'
5) At the very bottom, the confusion matrix for the Decision Tree model will be plotted.

(We did not plot the Neural Network confusion matrix as matplotlib opens another window for the plot, and this opened plot messes with the runtime measurement of our script)

While running the file, NOTICE 1) Comments. We kept helper methods for some functions from towardsdatascience.com and geeksforgeeks so as to aid us if needed.
However, those helper functions (if commented) are not used. The archive/ folder contains the dataset for the Fake/Real news that our model is trained on.

Comments explaining the code flowthrough are included. Our script runs Neural Network and Decision Tree classifers. Libraries used are specified in the requirements.txt file.
