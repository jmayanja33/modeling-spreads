from setuptools import setup, find_packages

version = "0.1"

requires = [
    "plotly",
    "dash",
    "dash_bootstrap_components",
    "pandas",
    "scikit-learn",
    "numpy",
    "matplotlib",
    "fastapi",
    "uvicorn",
    "pydantic",
    "requests",
    "joblib",
    "nfl-data-py",
]

config = {
    "name": "cse6242_project",
    "version": version,
    "author": "Team 176",
    "install_requires": requires,
    "python_requires": ">=3.10",
    "packages": find_packages(),
    "description": "Bet smarter, beat the bookie.",
    "include_package_data": True,
    "entry_points": {
        "console_scripts": ["start-beating-the-bookie=cse6242_project.app:run_cli"]
    },
}

setup(**config)
