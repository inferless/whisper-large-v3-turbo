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
        audio_url = inputs["audio_url"]
        return_timestamps = inputs.get("return_timestamps",False)
        max_new_tokens = inputs.get("max_new_tokens",None)
        language = inputs.get("language",None)
        task = inputs.get("task",None)
        temperature = inputs.get("temperature",None)
        
        result = self.pipe(audio_url,
              return_timestamps=return_timestamps,
              generate_kwargs={"max_new_tokens": max_new_tokens,
                               "language": language,
                               "task": task,
                               "temperature": temperature,
                              }
        )
        
        if not return_timestamps:
            return {"output": result["text"]}
        else:
            from_timestamp = []
            to_timestamp = []
            chunk_text = []

            for chunk in result['chunks']:
                print(chunk['timestamp'])
                print(chunk['text'])
                from_timestamp.append(chunk['timestamp'][0])
                to_timestamp.append(chunk['timestamp'][1])
                chunk_text.append(chunk['text'])

            return {
                "output":result["text"],
                "from_timestamp": from_timestamp,
                "to_timestamp":to_timestamp,
                "chunk_text":chunk_text
                }

    def finalize(self):
        self.pipe = None
