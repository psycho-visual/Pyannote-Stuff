import download, os, json, subprocess
import diarization, whisper, processing
from WORK_ON_THIS_ONE import final_transcript

RTTM_PATH = "audio.rttm"

def to_wav(audio_path):
    # convert anything (mp3, m4a, etc.) to 16kHz mono wav for pyannote
    root, _ = os.path.splitext(audio_path)
    wav_path = root + "_16k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", wav_path],
        check=True,
        capture_output=True,
    )
    return wav_path

def process_audio(AUDIO_PATH):
    AUDIO_PATH = to_wav(AUDIO_PATH)   # <-- add this one line
    print("running diarization...")
    annotation = diarization.diarize_audio(AUDIO_PATH)
    with open(RTTM_PATH, "w") as rttm:
        annotation.write_rttm(rttm)

    # this pulls some data out of the rttm
    print("parsing...")
    onset_list, duration_list, speaker_list = diarization.parse_rttm(RTTM_PATH)

    # timestamps to split the audio at to fit whisper's 25mb limit
    print("getting timestamps to split at...")
    timestamp_list = processing.get_split_timestamps(onset_list, duration_list)

    print("resampling and splitting audio...")
    segment_temp_dir, segment_temp_file_list = processing.segment_resample_audio(AUDIO_PATH, timestamp_list)

    print("processing filenames...")
    segment_file_list = whisper.combine_segment_filenames(segment_temp_dir, segment_temp_file_list)

    print("running segments through whisper...")
    temp_transcript_list = whisper.process_all_segments(segment_file_list)

    print("combining transcripts...")
    full_transcription_list = whisper.combine_transcripts(timestamp_list, temp_transcript_list)
    with open("full_transcription.json", 'w') as file:
        json.dump(full_transcription_list, file, indent=2)

    combined_diarization_list = diarization.combine_authors(onset_list, duration_list, speaker_list)

    print("assembling final transcript...")
    return final_transcript(full_transcription_list, combined_diarization_list)


def run_interview(url):
    print("downloading audio...")
    dir, AUDIO_PATH = download.download_audio_only(url, save_to_disk=True)
    return process_audio(AUDIO_PATH)


if __name__ == '__main__':
    run_interview("https://www.youtube.com/watch?v=t9ufyuX4Cho")