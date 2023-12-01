Note: Python version 3.10 is required to run.

1.) Project Description
cse6242_project is a Python library for helping you make smarter, more informed NFL bets.
Once started, the dashboard will allow you to select a team from a dropdown list a the top of the page.
The predicted score for the selected team, and the opponent team will be predicted using a random forest.
The team scores are displayed, along with the total score, spread, and distribution of predictions provided by the model.
The most decisive team stats for the predictions are displayed in a side-by-side bar chart for easy comparison.
All of this information displayed at your fingertips will help you make more informed betting decisions and help you beat the bookie.


2.) Installation
CLI Setup Steps:
1) Download source code using command "git clone https://github.com/jmayanja33/modeling-spreads.git"
2) Move to cloned directory "cd modeling-spreads"
3) Create python virtual environment using  "python -m venv venv"
4) Source the new environment using appropriate setup script in your virual environment.
    - for linux:  "source ./venv/bin/activate"
    - for windows powershell:  ".\venv\Scripts\Activate.ps1"
5) Install the package using command "pip install ."


3.) Execution
Start the dashboard using cli command "start-beating-the-bookie"
You can specify host and port using --host \<str> and --port \<int>.
The default host is 127.0.0.1
The default port is 8000

4.) Demo Video
Link to a installation demo video: https://www.youtube.com/watch?v=x4AI9Y-FhKk
