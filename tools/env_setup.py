import os
import shutil
from dotenv import load_dotenv
load_dotenv()
FFMPEG_PATH = r"C:\ffmpeg\ffmpeg-8.0.1-essentials_build\bin"
#os.environ["LITELLM_DEFAULT_PROVIDER"] = "groq"

#os.environ.pop("OPENAI_API_KEY", None)

def setup_environment():
    # Add FFmpeg to PATH if not already present
    if shutil.which("ffmpeg") is None:
        os.environ["PATH"] += f";{FFMPEG_PATH}"

    # Final validation
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg not found. Please install FFmpeg and add it to PATH."
        )
