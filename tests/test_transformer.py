from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tests.dummy_datasets import (
    DummyImageDataset,
    DummyTextDataset,
    DummyAudioDataset,
    DummyMultimodalDataset,
)
from engines.factory import Factory
from engines.trainer import ViTTrainer, TextTrainer, AudioTrainer, MultimodalTrainer, StandardTrainer
from engines.tester import ViTTester, TextTester, AudioTester, MultimodalTester, StandardTester
from models.transformer.model import TransformerModel


def test_transformer_model_modalities() -> None:
    # 1. Vision (image)
    img_model = TransformerModel(modality="image", in_channels=3, img_size=16, patch_size=4, embed_dim=32, num_heads=2, num_layers=1, num_classes=5)
    x_img = torch.randn(2, 3, 16, 16)
    out_img = img_model(x_img)
    assert out_img.shape == (2, 5)

    # 2. Text
    text_model = TransformerModel(modality="text", vocab_size=100, embed_dim=32, num_heads=2, num_layers=1, num_classes=3)
    x_text = torch.randint(0, 100, (2, 16))
    out_text = text_model(x_text)
    assert out_text.shape == (2, 3)

    # 3. Audio
    audio_model = TransformerModel(modality="audio", in_channels=1, freq_bins=16, patch_size=4, embed_dim=32, num_heads=2, num_layers=1, num_classes=4)
    x_audio = torch.randn(2, 1, 16, 16)
    out_audio = audio_model(x_audio)
    assert out_audio.shape == (2, 4)

    # 4. Multimodal dict batch
    multi_model = TransformerModel(modality="multimodal", in_channels=3, img_size=16, patch_size=4, vocab_size=100, embed_dim=32, num_heads=2, num_layers=1, num_classes=5)
    batch_multi = {"image": torch.randn(2, 3, 16, 16), "text": torch.randint(0, 100, (2, 16))}
    out_multi = multi_model(batch_multi)
    assert out_multi.shape == (2, 5)


def test_vit_trainer_and_tester() -> None:
    model = TransformerModel(modality="image", in_channels=3, img_size=16, patch_size=4, embed_dim=32, num_heads=2, num_layers=1, num_classes=5)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()

    trainer = ViTTrainer(model=model, optimizer=opt, criterion=crit)
    dataset = DummyImageDataset(num_samples=8, image_size=16, num_classes=5)
    loader = DataLoader(dataset, batch_size=4)

    trainer.fit(loader, num_epochs=1)
    assert trainer.current_epoch == 1

    tester = ViTTester(model=model, criterion=crit)
    metrics = tester.test(loader)
    assert "loss" in metrics


def test_text_trainer_and_tester() -> None:
    model = TransformerModel(modality="text", vocab_size=100, embed_dim=32, num_heads=2, num_layers=1, num_classes=3)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()

    trainer = TextTrainer(model=model, optimizer=opt, criterion=crit)
    dataset = DummyTextDataset(num_samples=8, vocab_size=100, seq_len=16, num_classes=3)
    loader = DataLoader(dataset, batch_size=4)

    trainer.fit(loader, num_epochs=1)
    assert trainer.current_epoch == 1

    tester = TextTester(model=model, criterion=crit)
    metrics = tester.test(loader)
    assert "loss" in metrics


def test_audio_trainer_and_tester() -> None:
    model = TransformerModel(modality="audio", in_channels=1, freq_bins=16, patch_size=4, embed_dim=32, num_heads=2, num_layers=1, num_classes=4)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()

    trainer = AudioTrainer(model=model, optimizer=opt, criterion=crit)
    dataset = DummyAudioDataset(num_samples=8, in_channels=1, freq_bins=16, time_steps=16, num_classes=4)
    loader = DataLoader(dataset, batch_size=4)

    trainer.fit(loader, num_epochs=1)
    assert trainer.current_epoch == 1

    tester = AudioTester(model=model, criterion=crit)
    metrics = tester.test(loader)
    assert "loss" in metrics


def test_multimodal_trainer_and_tester() -> None:
    model = TransformerModel(modality="multimodal", in_channels=3, img_size=16, patch_size=4, vocab_size=100, embed_dim=32, num_heads=2, num_layers=1, num_classes=5)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()

    trainer = MultimodalTrainer(model=model, optimizer=opt, criterion=crit)
    dataset = DummyMultimodalDataset(num_samples=8, image_size=16, vocab_size=100, seq_len=16, audio_bins=16, audio_steps=16, num_classes=5)
    loader = DataLoader(dataset, batch_size=4)

    trainer.fit(loader, num_epochs=1)
    assert trainer.current_epoch == 1

    tester = MultimodalTester(model=model, criterion=crit)
    metrics = tester.test(loader)
    assert "loss" in metrics


def test_factory_build_modality_trainers_testers() -> None:
    factory = Factory()
    model = TransformerModel(modality="image", in_channels=3, img_size=16, patch_size=4, embed_dim=32, num_heads=2, num_layers=1, num_classes=5)

    for trainer_name, expected_cls in [
        ("vit", ViTTrainer),
        ("text", TextTrainer),
        ("audio", AudioTrainer),
        ("multimodal", MultimodalTrainer),
        ("standard", StandardTrainer),
    ]:
        cfg = {
            "name": trainer_name,
            "model_instance": model,
            "optimizers": {"optimizer": {"type": "Adam", "lr": 0.001}},
            "criteria": {"main": {"type": "cross_entropy"}},
        }
        tr = factory.build_trainer(cfg)
        assert isinstance(tr, expected_cls)

    for tester_name, expected_cls in [
        ("vit", ViTTester),
        ("text", TextTester),
        ("audio", AudioTester),
        ("multimodal", MultimodalTester),
        ("standard", StandardTester),
    ]:
        cfg = {
            "name": tester_name,
            "model_instance": model,
            "criteria": {"main": {"type": "cross_entropy"}},
        }
        te = factory.build_tester(cfg)
        assert isinstance(te, expected_cls)


def test_transformer_edge_cases_and_uncovered_branches() -> None:
    # 1. 1D Audio Waveform (is_spectrogram=False)
    audio_1d_model = TransformerModel(modality="audio", is_spectrogram=False, in_channels=1, embed_dim=32, num_heads=2, num_layers=1, num_classes=4)
    x_1d = torch.randn(2, 1, 100)
    out_1d = audio_1d_model(x_1d)
    assert out_1d.shape == (2, 4)

    # 2. Multimodal batch with audio
    multi_model = TransformerModel(modality="multimodal", in_channels=3, vocab_size=100, is_spectrogram=False, embed_dim=32, num_heads=2, num_layers=1, num_classes=5)
    batch_with_audio = {
        "image": torch.randn(2, 3, 16, 16),
        "text": torch.randint(0, 100, (2, 16)),
        "audio": torch.randn(2, 1, 100),
    }
    out_multi_audio = multi_model(batch_with_audio)
    assert out_multi_audio.shape == (2, 5)

    # 3. Multimodal empty batch error
    with pytest.raises(ValueError, match="must be passed as a dictionary"):
        multi_model(torch.randn(2, 3))

    with pytest.raises(ValueError, match="must contain at least one"):
        multi_model({"unknown_key": torch.randn(2, 3)})

    # 4. Unknown modality error
    with pytest.raises(ValueError, match="Unknown modality"):
        invalid_model = TransformerModel(modality="invalid_modality")
        invalid_model(torch.randn(2, 3))

    # 5. Tester without labels returns output_mean
    img_model = TransformerModel(modality="image", embed_dim=32, num_heads=2, num_layers=1, num_classes=5)
    vit_tester = ViTTester(model=img_model)
    res_vit = vit_tester.test_step(torch.randn(2, 3, 32, 32))
    assert "output_mean" in res_vit

    text_model = TransformerModel(modality="text", vocab_size=100, embed_dim=32, num_heads=2, num_layers=1, num_classes=3)
    txt_tester = TextTester(model=text_model)
    res_txt = txt_tester.test_step({"text": torch.randint(0, 100, (2, 16))})
    assert "output_mean" in res_txt

    aud_tester = AudioTester(model=audio_1d_model)
    res_aud = aud_tester.test_step({"audio": torch.randn(2, 1, 100)})
    assert "output_mean" in res_aud

    multi_tester = MultimodalTester(model=multi_model)
    res_multi = multi_tester.test_step(batch_with_audio)
    assert "output_mean" in res_multi

