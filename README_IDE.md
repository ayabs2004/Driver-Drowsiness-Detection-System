# How to Fix "Import Errors" in VS Code

If you see red underlines under `import cv2`, `import numpy`, etc., but the app runs fine with `run_app.bat`, your IDE is simply not looking at the project's virtual environment.

Follow these steps to fix all red underlines:

1.  **Open VS Code** in your project folder (`driver ai project`).
2.  **Press `Ctrl + Shift + P`** (or `Cmd + Shift + P` on Mac) to open the Command Palette.
3.  Type **"Python: Select Interpreter"** and select it.
4.  Look for an option that points to **`.\.venv\Scripts\python.exe`**.
    *   It should say something like `Python 3.x.x ('.venv': venv)`.
5.  **Select that interpreter.**
6.  Wait a few seconds for the IDE to re-scan. **All red underlines should disappear.**

---

### Still seeing errors?
Run the diagnostic script to confirm your environment is healthy:
1.  Open a terminal in the project folder.
2.  Run: `.\.venv\Scripts\python.exe diagnose_imports.py`
3.  If the script says **"DIAGNOSIS: Everything is correctly installed"**, then your code is 100% fine and safe to run!
