# AdEval (Autonomous Driving Evaluator)
> The goal of the project is to compare multiple autonomous driving models against each other in various tasks

# Development Setup
1. Clone the repository
```
git clone https://github.com/jiabaoliu-015/CITS3200 && cd CITS3200
```

2. Create a virtual environment in the project directory
```
python -m venv .venv
```

3. Activate the virtual environment 
#### Windows
```
.venv\Scripts\activate
```
#### Mac
```
source .venv/bin/activate
```

4. Install libraries and makes it into a module (Just once unless new packages are added)
```
pip install --upgrade pip && pip install -e ".[dev]"
```

5. Run file
```
python -m adeval
```
