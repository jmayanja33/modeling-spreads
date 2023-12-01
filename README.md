Note: Python version 3.10 is required to run.
you can check your python version using:
```bash
python -V
```

# cse6242_project (Beat the Bookie)

cse6242_project is a Python library for helping you make smarter, more informed NFL bets.
Once started, the dashboard will allow you to select a team from a dropdown list a the top of the page.
The predicted score for the selected team, and the opponent team will be predicted using a random forest.
The team scores are displayed, along with the total score, spread, and distribution of predictions provided by the model.
The most decisive team stats for the predictions are displayed in a side-by-side bar chart for easy comparison.
All of this information displayed at your fingertips will help you make more informed betting decisions and help you beat the bookie.

## Installation
Highly recommended to use a [virtual environment](https://docs.python.org/3/library/venv.html).

```bash
python -m venv venv
```
Source your environment using the appropriate setup script.

Linux:
```bash
source ./venv/bin/activate
```

Windows Powershell:

```powershell
.\venv\Scripts\Activate.ps1
```

Clone the package from git and cd into the new directory.

```bash
git clone https://github.com/jmayanja33/modeling-spreads.git
cd modeling-spreads
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install cse6242_project.

```bash
pip install .
```

## Usage
Start dashboard application.

```bash
start-beating-the-bookie
```

You can specify host and port using --host \<str> and --port \<int>.

Default host: 127.0.0.1

Default port: 8000
```bash
start-beating-the-bookie --host 0.0.0.0 --port 9000
```

## Troubleshooting
**Note:** If you are experiencing issues building wheels on MacOS for python-snappy during the ```bash pip install .``` step, make sure to:
1) Update pip:
```bash
pip install --upgrade pip
```
2) Install snappy using [homebrew](https://brew.sh/):
```bash
brew install snappy  
CPPFLAGS="-I/usr/local/include -L/usr/local/lib" pip install python-snappy
```
3) Once done, you should be able to return to the "Usage" section to begin launching the app.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Authors and acknowledgment
Joshua McLennan Mayanja

Aditya Thakur

Sameer Rau

Kevin Buck

Ryan Buck
