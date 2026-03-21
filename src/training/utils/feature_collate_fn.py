import torch
from torch.nn.utils.rnn import pad_sequence


def whisper_feature_collate_fn(batch: list[dict]) -> dict:
    """
    Collate function for Whisper feature dataset.

    Why this function is needed:
    - audio feature lengths may differ
    - label token lengths definitely differ
    - batching requires consistent shapes

    This function pads:
    - input_features along the time dimension
    - labels along the token dimension
    """

    # Collect tensors
    input_features_list = [item["input_features"] for item in batch]
    labels_list = [item["labels"] for item in batch]

    # Find max time length among input features
    max_feature_length = max(feature.shape[-1] for feature in input_features_list)

    padded_input_features = []
    for feature in input_features_list:
        pad_amount = max_feature_length - feature.shape[-1]

        if pad_amount > 0:
            # Pad only the last dimension (time dimension)
            padded_feature = torch.nn.functional.pad(feature, (0, pad_amount))
        else:
            padded_feature = feature

        padded_input_features.append(padded_feature)

    # Stack padded input features into one batch tensor
    input_features = torch.stack(padded_input_features)

    # Pad labels with tokenizer pad value temporarily
    labels = pad_sequence(labels_list, batch_first=True, padding_value=50257)

    # Replace padding tokens with -100
    # Why:
    # In Hugging Face seq2seq training, label padding should usually be -100
    # so the loss function ignores padded positions.
    labels = labels.masked_fill(labels == 50257, -100)

    transcripts = [item["transcript"] for item in batch]
    languages = [item["language"] for item in batch]
    splits = [item["split"] for item in batch]
    standardized_audio_paths = [item["standardized_audio_path"] for item in batch]

    return {
        "input_features": input_features,
        "labels": labels,
        "transcripts": transcripts,
        "languages": languages,
        "splits": splits,
        "standardized_audio_paths": standardized_audio_paths,
    }