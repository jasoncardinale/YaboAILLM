import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import functional as F
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.datasets import load_tts_samples
from TTS.tts.configs.shared_configs import BaseDatasetConfig

if __name__ == "__main__":
    # Set the device to MPS if available, otherwise fallback to CPU
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    dataset_path = "./ljspeech_dataset"
    output_path = "./output"
    os.makedirs(output_path, exist_ok=True)

    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="metadata.csv",
        path=dataset_path,
    )

    config = GlowTTSConfig(
        batch_size=16,
        eval_batch_size=8,
        num_loader_workers=2,
        num_eval_loader_workers=2,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=10,
        text_cleaner="phoneme_cleaners",
        use_phonemes=True,
        phoneme_language="en-us",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        print_step=10,
        mixed_precision=False,  # MPS does not support mixed precision
        output_path=output_path,
        datasets=[dataset_config],
    )

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    print(f"Loaded {len(train_samples)} training samples and {len(eval_samples)} evaluation samples.")

    model = GlowTTS(config, ap, tokenizer).to(device)

    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    train_loader = DataLoader(train_samples, batch_size=config.batch_size, shuffle=True, num_workers=2)
    eval_loader = DataLoader(eval_samples, batch_size=config.eval_batch_size, shuffle=False, num_workers=2)

    for epoch in range(config.epochs):
        print(f"Epoch {epoch + 1}/{config.epochs}")
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            text = batch["text"]
            audio_file = batch["audio_file"]

            token_ids_batch = [torch.tensor(tokenizer.encode(t), dtype=torch.long).to(device) for t in text]

            spectrogram_batch = [
                torch.tensor(ap.audio_to_mel(ap.load_wav(a)), dtype=torch.float32).to(device) for a in audio_file
            ]

            token_ids_padded = torch.nn.utils.rnn.pad_sequence(token_ids_batch, batch_first=True).to(device)
            spectrogram_padded = torch.nn.utils.rnn.pad_sequence(spectrogram_batch, batch_first=True).to(device)

            token_lengths = torch.tensor([len(t) for t in token_ids_batch], dtype=torch.long).to(device)
            spectrogram_lengths = torch.tensor([len(s) for s in spectrogram_batch], dtype=torch.long).to(device)

            outputs = model(token_ids_padded, token_lengths, spectrogram_padded, spectrogram_lengths)
            loss = criterion(outputs, spectrogram_padded)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % config.print_step == 0:
                print(f"Step {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            eval_loss = 0.0
            for batch in eval_loader:
                text = batch["text"]
                audio_file = batch["audio_file"]

                token_ids_batch = [torch.tensor(tokenizer.encode(t), dtype=torch.long).to(device) for t in text]
                spectrogram_batch = [
                    torch.tensor(ap.audio_to_mel(ap.load_wav(a)), dtype=torch.float32).to(device) for a in audio_file
                ]

                token_ids_padded = torch.nn.utils.rnn.pad_sequence(token_ids_batch, batch_first=True).to(device)
                spectrogram_padded = torch.nn.utils.rnn.pad_sequence(spectrogram_batch, batch_first=True).to(device)

                token_lengths = torch.tensor([len(t) for t in token_ids_batch], dtype=torch.long).to(device)
                spectrogram_lengths = torch.tensor([len(s) for s in spectrogram_batch], dtype=torch.long).to(device)

                outputs = model(token_ids_padded, token_lengths, spectrogram_padded, spectrogram_lengths)
                loss = criterion(outputs, spectrogram_padded)
                eval_loss += loss.item()

            eval_loss /= len(eval_loader)
            print(f"Epoch {epoch + 1} Evaluation Loss: {eval_loss:.4f}")

    print("Training complete!")
