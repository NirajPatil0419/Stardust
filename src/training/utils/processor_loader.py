from transformers import WhisperProcessor
from config.model_config import WHISPER_MODEL_NAME


# Global model identifier
# Why global:
# We want one central place to control which processor/model family is used.
# Later, if you want to switch from whisper-small to whisper-base or whisper-medium,
# you only change it here instead of changing many files.
WHISPER_MODEL_NAME = "openai/whisper-small"


def load_whisper_processor(model_name: str = WHISPER_MODEL_NAME) -> WhisperProcessor:
    """
    Load the Hugging Face Whisper processor.

    Why this function is needed:
    Whisper does not directly take raw transcript strings or plain waveforms in training code.
    The processor handles:
    - audio feature extraction
    - text tokenization

    Parameters:
        model_name: Hugging Face model identifier

    Returns:
        WhisperProcessor instance
    """
    processor = WhisperProcessor.from_pretrained(model_name)
    return processor