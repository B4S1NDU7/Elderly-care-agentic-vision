# Run Instructions

This guide takes a new user from a fresh clone to a generated activity report.
Commands are shown for Windows PowerShell first, with macOS/Linux setup where
it differs. Run commands from the repository root unless noted otherwise.

## 1. Prerequisites

- Git
- Python 3.10 or 3.11 recommended
- An OpenAI API key for live video analysis only
- Internet access to install packages; the first pose-enabled run may also
  download YOLO model weights

The offline mock demo does not need an API key.

## 2. Clone and install

### Windows PowerShell

```powershell
git clone https://github.com/B4S1NDU7/Elderly-care-agentic-vision.git
Set-Location .\Elderly-care-agentic-vision

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `py -3.11` is unavailable, install Python 3.10 or 3.11 and use its
matching launcher, such as `py -3.10`. Running the virtual environment's
Python directly avoids PowerShell execution-policy problems with activation.

### macOS / Linux

```bash
git clone https://github.com/B4S1NDU7/Elderly-care-agentic-vision.git
cd Elderly-care-agentic-vision
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Run the offline demo

This creates a deterministic synthetic 60-second video, analyzes it without
OpenAI calls, and writes reports to `demo_output/`.

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe .\demo.py `
  --duration 60 `
  --mock `
  --interval 1 `
  --output-dir .\demo_output
```

macOS / Linux (with the virtual environment activated):

```bash
python demo.py --duration 60 --mock --interval 1 --output-dir demo_output
```

The generated `demo_video.mp4` is a simple synthetic test scene. Its
classification results are only a pipeline smoke test, not evidence of
real-world activity-recognition accuracy.

## 4. Analyze your own video

### Live model (requires an OpenAI API key)

Create a local `.env` file from the example:

Windows PowerShell:

```powershell
Copy-Item .env.example .env
notepad .env
```

macOS / Linux:

```bash
cp .env.example .env
```

Edit `.env` and set:

```text
OPENAI_API_KEY=your_openai_api_key
```

Keep `.env` private; do not commit or share it. Then analyze a video:

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe .\main.py `
  --video .\path\to\your_video.mp4 `
  --output .\output `
  --interval 2
```

macOS / Linux:

```bash
python main.py --video path/to/your_video.mp4 --output output --interval 2
```

The default sampling interval is two seconds. A shorter interval analyzes
more frames and may improve temporal detail at the cost of runtime and API
usage. The pose detector is enabled by default; pass `--no-pose` to disable
it, or `--no-agent` to disable agentic temporal review.

### Offline analysis of an existing video

To run without an API key, add `--mock`:

```powershell
.\.venv\Scripts\python.exe .\main.py `
  --video .\path\to\your_video.mp4 `
  --mock `
  --output .\output
```

Mock mode is for offline testing and does not replace live vision analysis.

## 5. Evaluate against ground truth (optional)

Ground truth must be manually annotated for the same video being analyzed.
Use [ground_truth_template.json](./ground_truth_template.json) as the schema
reference, save your annotations as a separate JSON file, and pass it with
`--gt`:

```powershell
.\.venv\Scripts\python.exe .\main.py `
  --video .\path\to\your_video.mp4 `
  --gt .\path\to\your_annotations.json `
  --output .\evaluation_output
```

For the bundled synthetic demo labels, first create the demo video as in
section 3, then run:

```powershell
.\.venv\Scripts\python.exe .\main.py `
  --video .\demo_video.mp4 `
  --mock `
  --interval 1 `
  --gt .\sample_output\synthetic_ground_truth.json `
  --output .\evaluation_output
```

The report includes activity accuracy and confusion matrix, per-state
duration error, and bed-exit/return precision, recall, and F1. Synthetic
evaluation results do not establish performance on real footage.

### Run the temporal stress scenarios

This runs six deterministic scripted state sequences through the temporal
tracker and writes per-scenario metrics plus failure examples. It requires no
video, API key, or additional dependency:

```powershell
.\.venv\Scripts\python.exe .\scenario_evaluation.py --output .\sample_output
```

The suite includes stand-and-return, correct UNKNOWN handling for blanket
occlusion, and three injected label-error examples (caregiver identity switch,
poor lighting, and camera-view loss). These test temporal tracking only; they
do not evaluate visual recognition. Results are saved to
`challenging_case_evaluation.json` and `failure_case_examples.md`.

## 6. Find the generated reports

The selected output directory contains:

- `activity_timeline.txt` — merged, timestamped activity segments
- `activity_summary.txt` and `activity_summary.json` — state and bed durations
- `bed_events.json` — detected bed exits and returns
- `full_report.json` — complete machine-readable result
- `agent_reasoning.txt` — agent decisions, when agentic review is enabled
- `evaluation_report.txt` — evaluation metrics when `--gt` is supplied

## 7. Run tests

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

On macOS / Linux with the virtual environment activated:

```bash
python -m unittest discover -s tests -v
```

## 8. Optional notebook

The command-line interface does not require Jupyter. To run the notebook,
install JupyterLab in the virtual environment and launch it from the
repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install jupyterlab
.\.venv\Scripts\python.exe -m jupyter lab
```

Open `demo.ipynb` and run its cells in order.
