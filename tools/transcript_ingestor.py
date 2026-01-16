import os
from faster_whisper import WhisperModel


class TranscriptIngestor:
    def __init__(self, model_size="base"):
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")


    def transcribe(self, video_path):
        segments, info = self.model.transcribe(video_path)
        transcript = []
        for seg in segments:
            transcript.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip()
            })
        return transcript