import torch
from torch.nn.utils.rnn import pad_sequence


def audio_collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate function for batching variable-length audio samples.

    Why this function is needed:
    Audio clips do not all have the same duration, so their waveform tensors
    have different lengths. PyTorch's default DataLoader collate function
    tries to stack tensors directly, which fails when shapes differ.

    This function:
    - collects waveform tensors from each sample
    - pads them to the same length
    - keeps transcript and metadata as lists

    Input:
        batch = list of dataset samples
        Each sample is a dictionary returned by AudioTranslationDataset

    Output:
        A dictionary containing:
        - padded_waveforms
        - waveform_lengths
        - sample_rates
        - transcripts
        - languages
        - splits
        - standardized_audio_paths
    """

    # Extract waveform tensors from each sample
    waveforms = [item["waveform"] for item in batch]

    # Store original waveform lengths before padding
    waveform_lengths = torch.tensor([len(waveform) for waveform in waveforms], dtype=torch.long)

    # Pad waveforms so they all have the same length
    # batch_first=True makes output shape: [batch_size, max_length]
    padded_waveforms = pad_sequence(waveforms, batch_first=True)

    # Keep metadata as lists
    sample_rates = [item["sample_rate"] for item in batch]
    transcripts = [item["transcript"] for item in batch]
    languages = [item["language"] for item in batch]
    splits = [item["split"] for item in batch]
    standardized_audio_paths = [item["standardized_audio_path"] for item in batch]

    return {
        "padded_waveforms": padded_waveforms,
        "waveform_lengths": waveform_lengths,
        "sample_rates": sample_rates,
        "transcripts": transcripts,
        "languages": languages,
        "splits": splits,
        "standardized_audio_paths": standardized_audio_paths,
    }