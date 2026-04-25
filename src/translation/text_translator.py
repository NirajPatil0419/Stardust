import sys
from transformers import MarianMTModel, MarianTokenizer


MODEL_NAME = "Helsinki-NLP/opus-mt-en-hi"


def load_translation_model():
    """
    Loads English-to-Hindi translation model.

    Why:
    This gives us a simple demo for non-technical users:
    English sentence in, Hindi sentence out.
    """
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME)
    return tokenizer, model


def translate_english_to_hindi(text: str) -> str:
    """
    Translate English text into Hindi.
    """
    tokenizer, model = load_translation_model()

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = model.generate(**inputs)

    translated_text = tokenizer.decode(
        translated_tokens[0],
        skip_special_tokens=True
    )

    return translated_text


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("English to Hindi Translator")
    print("Type 'exit' to stop.\n")

    while True:
        text = input("Enter English sentence: ")

        if text.lower().strip() == "exit":
            print("Translator stopped.")
            break

        hindi_text = translate_english_to_hindi(text)

        print(f"Hindi translation: {hindi_text}\n")


if __name__ == "__main__":
    main()