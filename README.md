# Sample Prepper

Cloud-based audio sample processor that normalizes, pitch-corrects, and trims audio files for use in samplers.

## Quick Start

1. Open the [Colab Notebook](https://colab.research.google.com/drive/1nOUHvvURPSNuj8DrHq0ECVAD395Q90ub?usp=sharing)
2. Run all cells
3. Upload an audio file through the generated API endpoint

## Features

- Amplitude normalization
- Silence trimming
- Pitch detection
- Pitch shift
- Simple REST API interface

## Demo

Run the Colab notebook's API cell to generate a temporary ngrok URL for live testing. During the presentation, I will provide the active URL.

## Technology Stack

- Google Colab (Cloud Runtime)
- CUDA
- Python 3.10
- Flask (REST API)
- ngrok (Tunneling)
- Firebase (Storage)

- Audio Analysis and Processing:
  - PyTorch
  - Torchaudio
  - Librosa
  - PyDub
  - FFMpeg

## Setup & Dependencies

All dependencies are automatically installed when running the Colab notebook.

Required packages:

```
python
flask
flask-cors
pyngrok
firebase-admin
torch
torchaudio
librosa
pydub
```

## API Usage

POST request to `/process` with a WAV file in the request body.
Returns a processed WAV file.

## Development

The project is actively being developed with plans to:

- Implement torchaudio processing pipeline
- Add batch processing capabilities
- Improve pitch detection accuracy

## Repository Structure

- `SamplePrepper_REI504M.ipynb`: Main Colab notebook containing the API implementation
- `report.pdf`: Project documentation
- `requirements.txt`: Python dependencies
- `dockerfile`: Dockerfile

## License

MIT
