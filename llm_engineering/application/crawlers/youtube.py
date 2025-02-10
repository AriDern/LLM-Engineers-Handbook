import whisper
import yt_dlp
from pyannote.audio import Pipeline
from collections import defaultdict
import os
import json
import sys
import subprocess
from urllib.parse import urlparse
from loguru import logger
from pytube import YouTube

from .base import BaseCrawler

from llm_engineering.domain.documents import YouTubeDocument


def delete_audio_file(file_path):
    """
    Deletes the specified audio file after processing.

    Parameters:
    - file_path (str): Path to the file to be deleted.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        else:
            print(f"File not found, skipping deletion: {file_path}")
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")


def fix_audio_format(input_path, output_path="fixed_audio.wav"):
    """
    Converts an audio file to a proper WAV format (PCM S16LE, 16 kHz, mono)
    to ensure compatibility with speaker diarization models.

    Parameters:
    - input_path (str): Path to the input audio file.
    - output_path (str): Path to save the converted WAV file.

    Returns:
    - output_path (str): Path to the properly formatted audio file.
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Error: File '{input_path}' does not exist.")

    print(f"Converting '{input_path}' to WAV format...")

    command = [
        "ffmpeg",
        "-y",  # Overwrite output if it exists
        "-i",
        input_path,  # Input file
        "-acodec",
        "pcm_s16le",  # Encode as PCM 16-bit little-endian
        "-ar",
        "16000",  # Resample to 16 kHz
        "-ac",
        "1",  # Convert to mono
        output_path,  # Output file
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Conversion successful! Saved as '{output_path}'.")
        return output_path  # Return the path of the fixed file
    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg conversion: {e.stderr.decode()}")
        raise


class YouTubeCrawler(BaseCrawler):
    """A class to extract, transcribe, and analyze YouTube video transcripts with speaker diarization."""

    model = YouTubeDocument

    def __init__(self, model_size="medium"):
        """Initialize the YouTube Crawler with required models."""
        self.hf_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
        self.model_size = model_size

    def download_audio(self, url, output_file="audio.wav"):
        """Download YouTube video as an audio file."""
        ydl_opts = {
            "format": "bestaudio/best",
            "extractaudio": True,
            "audioformat": "wav",
            "outtmpl": output_file,
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-us,en;q=0.5",
                "Accept-Encoding": "gzip,deflate",
                "Accept-Charset": "ISO-8859-1,utf-8;q=0.7,*;q=0.7",
            },
            "verbose": True,  # This will help with debugging
            "quiet": False,
            "no_warnings": False,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_file

    def transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper."""
        return self.whisper_model.transcribe(audio_path, word_timestamps=True)

    def diarize_audio(self, audio_path):
        """Perform speaker diarization on the audio file."""
        return self.diarization_pipeline(audio_path)

    def match_transcript_with_speakers(self, transcript, diarization):
        """Assign speaker labels to the transcript based on timestamps."""
        speaker_transcript = []
        speaker_dialogues = defaultdict(list)

        for segment in transcript["segments"]:
            start, end, text = segment["start"], segment["end"], segment["text"]
            assigned_speaker = "Unknown"

            for diarization_segment, _, speaker in diarization.itertracks(yield_label=True):
                if diarization_segment.start <= start <= diarization_segment.end:
                    assigned_speaker = speaker
                    break

            speaker_transcript.append(f"[{start:.2f}s - {end:.2f}s] {assigned_speaker}: {text}")
            speaker_dialogues[assigned_speaker].append(text)

        return speaker_transcript, speaker_dialogues

    def get_most_frequent_speaker(self, speaker_dialogues):
        """Find the speaker with the most dialogue."""
        most_speaker = max(speaker_dialogues, key=lambda spk: len(" ".join(speaker_dialogues[spk])))
        return most_speaker, " ".join(speaker_dialogues[most_speaker])

    def process_video(self, url):
        """Complete pipeline: download audio, transcribe, diarize, and match speakers."""
        audio_file = "audio.wav"

        print("Downloading YouTube audio...")
        self.download_audio(url, audio_file)

        print("Fixing audio file...")
        fixed_audio = fix_audio_format(audio_file)

        print("Transcribing audio with Whisper...")
        transcript = self.transcribe_audio(fixed_audio)

        print("Performing speaker diarization with Pyannote...")
        diarization = self.diarize_audio(fixed_audio)

        print("Matching speakers to transcript...")
        _, speaker_dialogues = self.match_transcript_with_speakers(transcript, diarization)

        print("Identifying the speaker who talks the most...")
        most_speaker, dialogue = self.get_most_frequent_speaker(speaker_dialogues)

        output = dialogue

        # with open("most_frequent_speaker.json", "w") as f:
        #    json.dump(output_json, f, indent=4)

        return output

    def extract(self, link: str, **kwargs) -> None:
        self.hf_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
        logger.info(f"Using HF Token: {self.hf_token}")
        logger.info(f"Using new HF Token: {self.hf_token}")
        self.model_size = "medium"
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=self.hf_token
        )
        self.whisper_model = whisper.load_model(self.model_size)

        old_model = self.model.find(link=link)
        if old_model is not None:
            logger.info(f"YouTube link already exists in the database: {link}")

            return

        logger.info(f"Starting extracting YouTube link: {link}")

        _content = self.process_video(link)

        content = {
            "Content": _content,
        }

        parsed_url = urlparse(link)
        platform = parsed_url.netloc

        user = kwargs["user"]
        instance = self.model(
            content=content,
            link=link,
            platform=platform,
            author_id=user.id,
            author_full_name=user.full_name,
        )
        instance.save()

        logger.info(f"Finished scrapping YouTube link: {link}")


# Example Usage
"""
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python youtube_crawler.py <YouTube_URL>")
        sys.exit(1)

    youtube_url = sys.argv[1]


    HF_TOKEN = #os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("Please set the HF_TOKEN environment variable with your Hugging Face API token.")

    crawler = YouTubeCrawler(hf_token=HF_TOKEN, model_size="medium")

    delete_audio_file("audio.wav")  # Deletes the original file
    delete_audio_file("fixed_audio.wav")  # Deletes the processed file after diarization

    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    result = crawler.process_video(youtube_url)
    print(json.dumps(result, indent=4))
"""
