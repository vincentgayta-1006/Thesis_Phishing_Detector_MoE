PHISHING DECTOR MOE SYSTEM - INSTALLATION & RUN GUIDE
For Windows & Linux


STEP 1: DOWNLOAD AND PREPARE FILES
Extract all files to a folder (e.g., "phishing_detector_moe")


STEP 2: SETUP VIRTUAL ENVIRONMENT

WINDOWS (Command Prompt or PowerShell):
python -m venv venv
venv\Scripts\activate

LINUX/MAC (Terminal):
python3 -m venv venv
source venv/bin/activate

✅ You should see (venv) at the start of your command prompt.


STEP 3: INSTALL DEPENDENCIES
pip install -r requirements.txt


STEP 4: VERIFY SETUP
python test_run.py
If you see "✓ Model loaded successfully!", you're ready!

RUNNING THE SYSTEM
⚠️ IMPORTANT: ALWAYS USE run.py, NOT main.py
ALWAYS ACTIVATE VIRTUAL ENVIRONMENT FIRST!

Windows (each time):
venv\Scripts\activate
python run.py [command]

Linux/Mac (each time):
source venv/bin/activate
python run.py [command]

AVAILABLE COMMANDS
1. INTERACTIVE MODE (Recommended for testing)
python run.py interactive
Type any message or URL to analyze

Commands:
sample - See test examples
demo - Run demonstrations
analyze - View statistics
exit or quit - End session


2. SINGLE PREDICTION
python run.py predict --text "Your message" --url "http://example.com"
Examples:
python run.py predict --text "URGENT! Verify account!" --url "http://login.tk"
python run.py predict --text "Meeting reminder"  # Text only
python run.py predict --url "http://site.xyz"    # URL only


3. BATCH PROCESSING (CSV files)
python run.py batch --input samples.csv --output results.csv

CSV format (samples.csv):
text,url
"Verify your account","http://login.tk"
"Meeting reminder","https://teams.microsoft.com"
"Congratulations!",""
"","http://suspicious.xyz"

4. DEMONSTRATION
python run.py demo
Runs 6 test cases to show how the system works.

5. ANALYZE STATISTICS
python run.py analyze --samples 100
Shows how often each expert contributes.