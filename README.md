# Sample Prepper

### Student Assignment for Cloud Computing course at HÍ (REI504M)

Cloud-based audio processing API for automating audio sample preparation (normalization, silence trimming, and pitch correction) for use in audio applications.

## Quick Start

Upload or record audio, process it, and A/B test the results!
[Live Demo GUI](https://sample-prepper-test-ui.vercel.app/)
[client gh-repo](https://github.com/KristinnRoach/test-client)

API as colab notebook: 
[Colab Notebook](https://colab.research.google.com/drive/1nOUHvvURPSNuj8DrHq0ECVAD395Q90ub?usp=sharing)

## Architecture

### Development Environment (GPU Optional)

- Google Colab notebook with optional GPU acceleration
- FastAPI backend
- Ngrok for endpoint exposure
- Firebase Storage for processed files

### Client Application

- Hosted on Vercel
- Simple interface for testing audio processing

## Features

- Audio normalization (peak normalization to [-1, 1])
- Intelligent silence trimming
- Pitch detection and correction
- Async processing with background tasks
- Cross-origin resource sharing (CORS)
- Error handling and logging

## Technology Stack

Cloud Services:

- Google Colab (PAAS)
- Firebase Storage
- Ngrok tunneling

Core Processing:

- PyTorch/Torchaudio
- Librosa (pitch detection)
- FFMPEG/SOX (audio processing)
- FastAPI/Uvicorn

## API Endpoints

`POST /process`

- Upload audio file with processing options
- Returns processed WAV file
- Options:
- normalize: amplitude normalization
- trim: silence trimming
- tune: pitch correction
- outputFormat: output file format
- saveToFirebase: optionally persist the data

`GET /process`

- Health check endpoint

## Dependencies

All dependencies are automatically installed when running the Colab notebook:

```python
fastapi
uvicorn
librosa
python-multipart
ffmpeg-python
aiofiles
soundfile
pyngrok
nest_asyncio
torch
torchaudio
firebase-admin
```

### Repository Structure

prepper-jsc.py: Implementation for deploying or local use
requirements.txt: Python dependencies
Dockerfile: for cloud deployment (not tested)
setup.sh: System configuration and dependency installation script

#### Main implementation as Colab notebook can be found here:

#### https://colab.research.google.com/drive/1nOUHvvURPSNuj8DrHq0ECVAD395Q90ub?usp=sharing
