from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import logging
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import librosa
import os
import time
import mimetypes

# Create fixed directories
BASE_DIR = Path.cwd()  # Current working directory
UPLOAD_DIR = BASE_DIR / 'uploads'
OUTPUT_DIR = BASE_DIR 

# Create directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def get_mime_type(path: str) -> tuple:
    """Get the mime type of the audio file.
    
    Args:
        path (str): Path to the audio file
        
    Returns:
        tuple: Mime type and encoding
    """
    mime_type = mimetypes.guess_type(path)
    logger.info(f"{path}: Mime type: {mime_type}")
    return mime_type

def get_output_format(input_format: str, requested_format: str = None) -> str:
    """Determine the output format based on input and requested format.
    
    Args:
        input_format (str): Original file format (e.g., 'mp3', 'webm')
        requested_format (str): Optional requested output format
        
    Returns:
        str: Output format to use
    """
    valid_formats = {'wav', 'mp3', 'webm'}
    
    if requested_format and requested_format.lower() in valid_formats:
        return requested_format.lower()
    
    # Default to input format, or wav if input format not supported
    return input_format.lower() if input_format.lower() in valid_formats else 'wav'


def normalize(waveform: torch.Tensor) -> torch.Tensor:
    """Peak normalize audio to range [-1, 1].
    
    Args:
        waveform (torch.Tensor): Input audio waveform
        
    Returns:
        torch.Tensor: Normalized waveform
    """
    input_peak = torch.abs(waveform).max()
    logger.info(f"[NORMALIZE] Input peak amplitude: {input_peak}")

    if input_peak > 0:
        normalized_waveform = waveform / input_peak
        output_peak = torch.abs(normalized_waveform).max()
        logger.info(f"[NORMALIZE] Output peak amplitude: {output_peak}")
        return normalized_waveform
    return waveform

def get_pitch_factor(original_pitch: float, target_pitch: float) -> float:
    """Calculate the factor needed to transpose from original pitch to target pitch.
    
    Args:
        original_pitch (float): Original pitch frequency in Hz
        target_pitch (float): Target pitch frequency in Hz
        
    Returns:
        float: Pitch adjustment factor
        
    Raises:
        ValueError: If original pitch is not positive
    """
    if original_pitch <= 0:
        raise ValueError("Original pitch must be positive")
    return target_pitch / original_pitch

def transpose_torch(waveform: torch.Tensor, sample_rate: int, factor: float) -> torch.Tensor:
    """Transpose audio by resampling.
    
    Args:
        waveform (torch.Tensor): Input audio waveform
        sample_rate (int): Original sample rate
        factor (float): Pitch adjustment factor
        
    Returns:
        torch.Tensor: Transposed waveform
        
    Raises:
        TypeError: If inputs are of wrong type
        ValueError: If factor is not positive
    """
    if not isinstance(factor, (int, float)):
        raise TypeError("Factor must be a number")
    if factor <= 0:
        raise ValueError("Factor must be positive")
    if not isinstance(sample_rate, int):
        raise TypeError("Sample rate must be an integer")
    if not torch.is_tensor(waveform):
        raise TypeError("Waveform must be a torch.Tensor")

    resample_rate = int(sample_rate / factor)
    resampler = T.Resample(
        orig_freq=sample_rate,
        new_freq=resample_rate,
        dtype=waveform.dtype
    )
    
    return resampler(waveform)

def trim_silence(waveform: torch.Tensor, threshold_db: float = -50.0, 
                min_length_ms: float = 50, sr: int = 44100) -> torch.Tensor:
    """Trim silence from start and end of audio using numpy for fast processing.
    
    Args:
        waveform (torch.Tensor): Input audio waveform
        threshold_db (float): Threshold in decibels below which audio is considered silence
        min_length_ms (float): Minimum length of audio segment in milliseconds
        sr (int): Sample rate of the audio
        
    Returns:
        torch.Tensor: Trimmed waveform
    """
    try:
        # Convert to numpy and ensure it's flat
        audio_np = waveform.cpu().numpy()
        if audio_np.ndim == 2:
            audio_np = audio_np.squeeze(0)  # Remove channel dimension
            
        # Calculate frame length
        frame_length = int(min_length_ms * sr / 1000)
        
        # Quick returns for short audio
        if len(audio_np) < frame_length or len(audio_np) // frame_length == 0:
            return waveform
            
        # Reshape audio into frames (faster than processing sample by sample)
        frames = audio_np[:len(audio_np) - (len(audio_np) % frame_length)]
        frames = frames.reshape(-1, frame_length)
        
        # Vectorized RMS energy calculation
        rms = np.sqrt(np.mean(np.square(frames), axis=1))
        db = 20 * np.log10(rms + 1e-8)
        
        # Find start and end points above threshold
        mask = db > threshold_db
        nonzero = np.nonzero(mask)[0]
        
        if len(nonzero) == 0:
            return waveform
        
        # Calculate trim points
        start = nonzero[0] * frame_length
        end = min((nonzero[-1] + 1) * frame_length, len(audio_np))
        
        # Convert back to torch tensor efficiently
        trimmed = torch.from_numpy(audio_np[start:end]).to(waveform.device)
        return trimmed.unsqueeze(0) if waveform.dim() == 2 else trimmed
        
    except Exception as e:
        logger.warning(f"Silence trimming failed: {str(e)}")
        return waveform
    

def get_main_pitch(audio_data, sr, min_note='C1', max_note='C7'):
    """Get the main pitch from audio data"""
    try:
        # Ensure input is numpy array
        if torch.is_tensor(audio_data):
            audio_data = audio_data.numpy()
        
        # Calculate pitch using PYIN algorithm
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_data,
            fmin=librosa.note_to_hz(min_note),
            fmax=librosa.note_to_hz(max_note),
            sr=sr,
            frame_length=2048,
            win_length=1024,
            hop_length=512
        )

        # Filter out unvoiced and low probability segments
        mask = voiced_flag & (voiced_probs > 0.6)
        f0_valid = f0[mask]

        if len(f0_valid) == 0:
            logger.warning("No valid pitch detected")
            return None, None, 0.0

        # Get the median frequency
        median_f0 = float(np.median(f0_valid))
        
        # Convert to note
        closest_note = librosa.hz_to_note(median_f0)
        note_freq = librosa.note_to_hz(closest_note)
        note = {'closest_note': closest_note, 'freq': note_freq}
        
        # Calculate confidence
        confidence = float(np.mean(voiced_probs[voiced_flag]))
        
        return median_f0, note, confidence

    except Exception as e:
        logger.error(f"Pitch detection failed: {str(e)}")
        raise
## ________________ MUCH FASTER BUT ONLY CONSIDERING a Specified time segment of the audio  ________________

def detect_pitch_optimized(audio_data, sr, min_note='C1', max_note='C7', analysis_ratio=0.2):
    """Get the main pitch from audio data
    
    Args:
        audio_data: Input audio array
        sr: Sample rate
        min_note: Minimum note to detect
        max_note: Maximum note to detect
        analysis_ratio: Ratio of total audio length to analyze (0.0 to 1.0)
    """

    try:
        # Ensure input is numpy array
        if torch.is_tensor(audio_data):
            audio_data = audio_data.numpy()
        
        # Downsample for pitch detection if sample rate is high
        if sr > 22050:
            target_sr = 22050
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        # Calculate segment length based on ratio
        total_length = len(audio_data)
        segment_len = int(total_length * analysis_ratio)  # Simply take ratio of total length
        
        # Calculate segment boundaries
        start_pos = total_length // 8 # Start from 1/8th of audio 
        end_pos = start_pos + segment_len
        
        # Take segment from middle of audio
        audio_segment = audio_data[start_pos:end_pos]
        
        logger.info(f"Analyzing {analysis_ratio*100:.1f}% of audio "
                   f"({len(audio_segment)/sr:.3f}s out of {total_length/sr:.3f}s total) "
                   f"from position {start_pos/sr:.3f}s to {end_pos/sr:.3f}s")
        

        # Calculate pitch using PYIN algorithm with optimized parameters
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_segment,
            fmin=librosa.note_to_hz(min_note),
            fmax=librosa.note_to_hz(max_note),
            sr=sr,
            frame_length=1024,  # Reduced from 2048
            win_length=512,     # Reduced from 1024
            hop_length=256      # Reduced from 512
        )

        # Filter out unvoiced and low probability segments
        mask = voiced_flag & (voiced_probs > 0.6)
        f0_valid = f0[mask]

        if len(f0_valid) == 0:
            logger.warning("No valid pitch detected")
            return None, None, 0.0

        # Get the median frequency
        median_f0 = float(np.median(f0_valid))
        
        # Convert to note
        closest_note = librosa.hz_to_note(median_f0)
        note_freq = librosa.note_to_hz(closest_note)
        note = {'closest_note': closest_note, 'freq': note_freq}
        
        # Calculate confidence
        confidence = float(np.mean(voiced_probs[voiced_flag]))
        
        return median_f0, note, confidence

    except Exception as e:
        logger.error(f"Pitch detection failed: {str(e)}")
        raise


def process_audio_file(file_path, options):
    """Process audio file with given options."""
    start_time = time.time()
    results = []

    try:
        # Load audio and log time
        t0 = time.time()
        waveform, sr = torchaudio.load(file_path)
        waveform = waveform.to(device)
        logger.info(f"Load time: {time.time() - t0:.3f}s")
        
        # Convert to mono if stereo
        if waveform.size(0) > 1:
            t0 = time.time()
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            logger.info(f"Mono conversion time: {time.time() - t0:.3f}s")
        
        # Normalize if requested
        if options['normalize']:
            t0 = time.time()
            max_val = torch.max(torch.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val
            logger.info(f"Normalization time: {time.time() - t0:.3f}s")
        
        # Trim silence if requested
        if options['trim']:
            t0 = time.time()
            waveform = trim_silence(waveform, threshold_db=-50.0, min_length_ms=50)
            logger.info(f"Silence trimming time: {time.time() - t0:.3f}s")
        
        # Pitch adjustment if requested
        if options['tune']:
            # Detect pitch
            t0 = time.time()
            audio_np = waveform[0].cpu().numpy()
            detected_pitch, note, confidence = detect_pitch_optimized(audio_np, sr)
            logger.info(f"Pitch detection time: {time.time() - t0:.3f}s")
            
            if detected_pitch and confidence > 0.6:
                C4_FREQ = 261.6255
                target = options.get('target_pitch', C4_FREQ)
                factor = get_pitch_factor(detected_pitch, target)
                
                logger.info(f"Pitch adjustment: detected={detected_pitch:.1f}Hz, "
                          f"target={target:.1f}Hz, factor={factor:.3f}")
                
                if 0.5 <= factor <= 2.0:
                    # Use only sox version for pitch shifting
                    t0 = time.time()
                    effects = [
                        ["speed", str(factor)],
                        ["rate", str(sr)]
                    ]
                    processed_waveform, processed_sr = torchaudio.sox_effects.apply_effects_tensor(
                        waveform.cpu(), sr, effects)
                    logger.info(f"Sox pitch shifting time: {time.time() - t0:.3f}s")
                    results.append(("sox", processed_waveform, processed_sr))

        if not results:  # If no processing was done return original waveform 
            results.append(("original", waveform, sr))
               
        logger.info(f"Total processing time: {time.time() - start_time:.3f}s")
        return results
        
    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}")
        raise

# Initialize FastAPI app

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # add my client when deploying
    allow_credentials=True,
    allow_methods=["*"], # restrict to needed methods when deploying
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,  # Cache preflight requests for 24 hours
)

@app.get("/")
async def root():
    return {"message": "Sample Prepper API running"}


@app.get("/process")
async def health_check():
    return {"status": "ok"}

# Create thread pool for CPU-intensive tasks
thread_pool = ThreadPoolExecutor(max_workers=4)

async def run_in_thread(func, *args):
    """Run CPU-intensive tasks in thread pool"""
    return await asyncio.get_event_loop().run_in_executor(thread_pool, func, *args)

@app.post("/process")
async def process_audio_endpoint(
    file: UploadFile = File(...),
    options: str = Form(default=None)  
):
    """Process audio file endpoint"""
    # Log incoming request details
    logger.info(f"Received file: {file.filename}")
    logger.info(f"Content-Type: {file.content_type}")
    logger.info(f"Received options: {options}")

    # File size limit
    MAX_FILE_SIZE = 35 * 1024 * 1024  # 35MB for now
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    # Default options
    process_options = {
        'normalize': False,
        'trim': False,
        'tune': False,
        'returnType': 'blob',
        'outputFormat': 'wav'  
    }

    logger.info(f"Received options string: {options}")
    
    if options:
        try:
            # Parse the options string
            user_options = json.loads(options)
            logger.info(f"Parsed options: {user_options}")
            process_options.update(user_options)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse options: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid options format: {e}")
    
    # Log final options being used
    logger.info(f"Using process options: {process_options}")
    
    # Create unique filename
    timestamp = int(time.time() * 1000)
    safe_filename = f"{timestamp}_{file.filename}"
    input_path = UPLOAD_DIR / safe_filename
    output_path = None

    try:
        # Save uploaded file
        t0 = time.time()
        with open(input_path, "wb") as f:
            f.write(contents)
        logger.info(f"File save time: {time.time() - t0:.3f}s")

        # Process audio in thread pool
        t0 = time.time()
        results = await run_in_thread(process_audio_file, str(input_path), process_options)
        logger.info(f"Processing time: {time.time() - t0:.3f}s")

        if not results:
            raise HTTPException(status_code=500, detail="No output files generated")

        # Save processed version
        t0 = time.time()
        output_filename = f"processed_{timestamp}_{file.filename}"
        output_path = OUTPUT_DIR / output_filename

        # Get the processed waveform (sox version)
        waveform, sr = results[0][1], results[0][2]  # Always use first result (sox)

        # Ensure correct format before saving
        waveform = waveform.cpu()
        if torch.max(torch.abs(waveform)) > 1.0:
            waveform = waveform / torch.max(torch.abs(waveform))

        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)

        # Save with explicit format
        torchaudio.save(
            str(output_path),
            waveform,
            sr,
            encoding="PCM_S",
            bits_per_sample=16
        )
        logger.info(f"File save time: {time.time() - t0:.3f}s")
        
        # Use FileResponse for streaming the file
        return FileResponse(
            path=output_path,
            filename=output_filename,
            media_type='audio/wav'
        )

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup in background task
        async def cleanup():
            await asyncio.sleep(1)  # Wait for file transfer to complete
            if input_path.exists():
                input_path.unlink()
            if output_path and output_path.exists():
                output_path.unlink()

        background_tasks = BackgroundTasks()
        background_tasks.add_task(cleanup)

if __name__ == "__main__":
    logger.info("Starting Sample Prepper server...")
    uvicorn.run("main:app", host="0.0.0.0", port=5001, workers=4, log_level="info")

