import os
import sys

# --- make sure we can import from ../src and project root ---

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.ASR_assemblyai import download_and_extract_audio, transcribe_audio_with_assemblyai


def main():
    # 1. Download video + extract WAV into data/
    video_path, audio_path = download_and_extract_audio(
        youtube_url="https://youtu.be/64643Op6WJo?si=_XedZxhmBW6xvTV0",
        video_filename="lecture.mp4",
        audio_filename="lecture.wav",
    )

    # 2. Transcribe using AssemblyAI
    transcript_data = transcribe_audio_with_assemblyai(audio_path)

    # 3. Pretty print transcript with timestamps
    print("\n========== SEGMENTS WITH TIMESTAMPS ==========")
    for seg in transcript_data["segments"]:
        start = seg["start"]
        end = seg["end"]
        speaker = seg["speaker"]
        text = seg["text"]

        start_str = f"{start:.2f}s" if start is not None else "?"
        end_str = f"{end:.2f}s" if end is not None else "?"

        if speaker is not None:
            print(f"[{start_str} -> {end_str}] Speaker {speaker}: {text}")
        else:
            print(f"[{start_str} -> {end_str}] {text}")


if __name__ == "__main__":
    main()
