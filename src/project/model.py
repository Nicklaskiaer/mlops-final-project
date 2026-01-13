import torch
import torch.nn as nn
import librosa
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional, cast
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

class HubertClassifier(nn.Module):
    """
    HuBERT-based Audio Classifier compatible with torch.nn.Module.
    Wraps Hugging Face's HubertForSequenceClassification.
    """

    def __init__(
        self, 
        model_name: str = "ntu-spml/distilhubert",
        num_labels: int = 8, 
        freeze_feature_encoder: bool = True
    ):
        super().__init__()
        self.target_sampling_rate = 16000
        
        # Load Config and Model
        self.hf_model = HubertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        
        # Load Feature Extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

        if freeze_feature_encoder:
            self.hf_model.freeze_feature_encoder()

    def forward(
        self, 
        input_values: torch.Tensor, 
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Standard forward pass.
        Returns: logits (Tensor) OR (loss, logits) if labels are provided.
        """
        input_values = input_values.to(self.hf_model.device)
        if labels is not None:
            labels = labels.to(self.hf_model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.hf_model.device)

        outputs = self.hf_model(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        if labels is not None:
            return outputs.loss, outputs.logits
        return outputs.logits

    def predict(self, audio_input: Union[str, Path, np.ndarray, torch.Tensor]) -> dict:
        """
        Inference method for a single audio file or array.
        """
        self.eval()
        
        # 1. Load and Preprocess Audio
        waveform = self._ensure_audio_format(audio_input)
        
        # 2. Prepare for Model
        inputs = self.feature_extractor(
            waveform, 
            sampling_rate=self.target_sampling_rate, 
            return_tensors="pt", 
            padding=True
        )
        
        input_values = inputs.input_values.to(self.hf_model.device)

        # 3. Inference
        with torch.no_grad():
            output = self.forward(input_values)
            
            # Explicitly ensure we are working with the logits Tensor
            if isinstance(output, tuple):
                logits = output[1]
            else:
                logits = output

            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # .item() returns a number, casting to int satisfies the indexer
            predicted_id = int(torch.argmax(probabilities, dim=-1).item())

        return {
            "predicted_label_idx": predicted_id,
            "confidence": probabilities[0][predicted_id].item(),
            "probabilities": probabilities[0].cpu().numpy().tolist()
        }

    def _ensure_audio_format(self, audio_input: Union[str, Path, np.ndarray, torch.Tensor]) -> np.ndarray:
        """Helper to load audio and resample to 16kHz."""
        waveform = None
        
        if isinstance(audio_input, (str, Path)):
            path = str(audio_input)
            waveform, sr = librosa.load(path, sr=self.target_sampling_rate)
        
        elif isinstance(audio_input, np.ndarray):
            waveform = audio_input
            
        elif isinstance(audio_input, torch.Tensor):
            waveform = audio_input.cpu().numpy()
            if waveform.ndim > 1:
                waveform = waveform.squeeze()

        if waveform is None:
            raise ValueError("Unsupported audio input format.")

        return waveform

if __name__ == "__main__":
    model = HubertClassifier(num_labels=8)
    fake_audio = torch.randn(1, 16000)
    
    # Test Forward
    logits = model(fake_audio)
    # Ensure logits is a Tensor for printing
    if isinstance(logits, tuple):
        logits = logits[1]
    print(f"Output Logits Shape: {logits.shape}")