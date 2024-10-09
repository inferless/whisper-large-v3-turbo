import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class InferlessPythonModel:
        
    def initialize(self):
        model_id = "openai/whisper-large-v3-turbo"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            device_map="cuda"
        )
        
        processor = AutoProcessor.from_pretrained(model_id)
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch.float16,
            return_timestamps=True
        )

    def infer(self, inputs):
        audio_url = inputs["audio_url"]
        result = self.pipe(audio_url)
        
        return {"transcribed_output":result["text"]}

    def finalize(self):
        self.pipe = None