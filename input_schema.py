INPUT_SCHEMA = {
    "audio_url": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["https://github.com/rbgo404/Files/raw/main/jeanNL.mp3"]
    },
    "return_timestamps": {
        'datatype': 'BOOL',
        'required': False,
        'shape': [1],
        'example': [True]
    },
    "return_timestamps": {
        'datatype': 'INT64',
        'required': False,
        'shape': [1],
        'example': [400]
    },
    "language": {
        'datatype': 'STRING',
        'required': False,
        'shape': [1],
        'example': ['english']
    },
    "task": {
        'datatype': 'STRING',
        'required': False,
        'shape': [1],
        'example': ['transcribe']
    },
    "temperature": {
        'datatype': 'FP64',
        'required': False,
        'shape': [1],
        'example': [0.5]
    }
}
