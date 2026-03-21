from pathlib import Path
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset


class AudioTranslationDataset(Dataset):
    """
    PyTorch Dataset for loading standardized audio and transcript text.

    Why this class is needed:
    PyTorch training loops do not read CSV files directly.
    They expect a Dataset object that can:
    - tell how many samples exist
    - return one sample at a time
    - provide input and label data in a structured way

    This class reads a standardized manifest and loads:
    - audio waveform
    - transcript text
    - metadata such as language and split
    """

    def __init__(self, manifest_path: str):
        """
        Initialize dataset using a manifest CSV file.

        Parameters:
            manifest_path (str): path to manifest_standardized.csv

        Why this is needed:
        The manifest is the source of truth for the training data.
        It tells us where the standardized audio files are and what
        transcript belongs to each file.
        """
        self.manifest_path = Path(manifest_path)

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)

        required_columns = {
            "language",
            "split",
            "audio_file",
            "audio_path",
            "transcript",
            "standardized_audio_path",
            "target_sample_rate",
            "target_channels",
        }

        missing_columns = required_columns - set(self.df.columns)
        if missing_columns:
            raise ValueError(
                f"Manifest is missing required columns: {missing_columns}"
            )

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Why needed:
        PyTorch calls this to know dataset size.
        """
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        """
        Return one training sample by index.

        Output:
            A dictionary containing:
            - waveform
            - sample_rate
            - transcript
            - language
            - split
            - standardized_audio_path

        Why dictionary output is useful:
        It is flexible and easy to extend later when we add:
        - tokenized transcript
        - features
        - durations
        - translation targets
        """
        row = self.df.iloc[index]

        audio_path = row["standardized_audio_path"]
        transcript = row["transcript"]
        language = row["language"]
        split = row["split"]

        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Read audio file
        waveform, sample_rate = sf.read(audio_path)

        # Convert waveform to torch tensor
        waveform_tensor = torch.tensor(waveform, dtype=torch.float32)

        return {
            "waveform": waveform_tensor,
            "sample_rate": sample_rate,
            "transcript": transcript,
            "language": language,
            "split": split,
            "standardized_audio_path": audio_path,
        }