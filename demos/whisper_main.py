import os
import sys

# --- make sure we can import from ../src and from project root ---

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.ASR_whisper import download_and_extract_audio, transcribe_audio_local_whisper


def main():
    # 1. Download video and extract audio into data/
    video_path, audio_path = download_and_extract_audio(
        youtube_url="https://youtu.be/64643Op6WJo?si=_XedZxhmBW6xvTV0",
        video_filename="lecture.mp4",
        audio_filename="lecture.wav",
    )

    # 2. Run Whisper locally to get transcript + timestamps
    transcript_data = transcribe_audio_local_whisper(audio_path)

    # 3. Show results
    print("\n========== FULL TRANSCRIPT ==========")
    print(transcript_data["text"])

    print("\n========== SEGMENTS WITH TIMESTAMPS ==========")
    for seg in transcript_data["segments"]:
        start = seg["start"]
        end = seg["end"]
        text = seg["text"]

        start_str = f"{start:.2f}s" if start is not None else "?"
        end_str = f"{end:.2f}s" if end is not None else "?"

        print(f"[{start_str} -> {end_str}] {text}")


if __name__ == "__main__":
    main()
