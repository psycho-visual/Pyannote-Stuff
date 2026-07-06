import os, subprocess
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def transcribe_audio(file_path):
    client = Groq(
    api_key = os.getenv("GROQ_API_KEY")
    )

    with open(file_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=file,
            model="whisper-large-v3",
            response_format="verbose_json",
            timestamp_granularities=['word'],
        )
        json_dict = transcription.model_dump()
        return json_dict

def combine_segment_filenames(temp_dir, file_list):
    segment_file_list = []
    for file_path in file_list:
        full_segment_path = os.path.join(temp_dir, file_path)
        segment_file_list.append(full_segment_path)
    return segment_file_list

def process_all_segments(file_list):
    transcript_list = []
    for file in file_list:
        transcript_list.append(transcribe_audio(file))
    return transcript_list

def combine_transcripts(timestamp_list, transcript_list):
    full_transcript_words = []
    for i, transcript in enumerate(transcript_list):
        current_transcript_words = transcript['words']
        current_timestamp = timestamp_list[i]
        for word in current_transcript_words:
            word['start'] += current_timestamp
            word['end'] += current_timestamp
        full_transcript_words += current_transcript_words
    return full_transcript_words
        # something to add it to a bigger json at the end
