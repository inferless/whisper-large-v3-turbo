import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import json

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
        # Extracting inputs with default values
        audio_url = inputs["audio_url"]
        return_timestamps = inputs.get("return_timestamps", "word") # Can be True, False, or "word"
        max_new_tokens = inputs.get("max_new_tokens")
        language = inputs.get("language")
        task = inputs.get("task") # Can be "transcribe" or "translate"
        temperature = inputs.get("temperature")

        # Convert return_timestamps if needed
        return_timestamps = return_timestamps == "True" if return_timestamps in ["True", "False"] else return_timestamps
        # Call the pipeline with necessary parameters
        result = self.pipe(
        audio_url,
        return_timestamps=return_timestamps,
        generate_kwargs={
            "max_new_tokens": max_new_tokens,
            "language": language,
            "task": task,
            "temperature": temperature,
            },
        )
        # Prepare the output based on the return_timestamps condition
        if not return_timestamps:
            return {"output_text": result["text"]}

        # Extract timestamps and chunk text
        from_timestamp, to_timestamp, chunk_text = zip(
            *[(chunk['timestamp'][0], chunk['timestamp'][1], chunk['text']) for chunk in result['chunks']]
        )

        return {
            "output_text": [result["text"]],
            "from_timestamp": list(from_timestamp),
            "to_timestamp": list(to_timestamp),
            "chunk_text": list(chunk_text),
        }


    def finalize(self):
        self.pipe = None
