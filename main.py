import download, os, json
import diarization, whisper, processing
from WORK_ON_THIS_ONE import final_transcript

RTTM_PATH = "audio.rttm"
AUDIO_PATH = "audio.wav"


#placeholder
# whisper.downscale_audio("audio.wav")

# if there's a leftover rttm from a previous run, check before redoing the slow part
run_diarization = True
if os.path.exists(RTTM_PATH):
    answer = input("there's already an audio.rttm -- download and diarize again? (y/n): ")
    run_diarization = answer.strip().lower() == "y"

if run_diarization:
    url = input("Enter url: ")
    print("downloading audio...")
    dir, AUDIO_PATH = download.download_audio_only(url, save_to_disk=True)

    print("running diarization...")
    annotation = diarization.diarize_audio(AUDIO_PATH)
    with open(RTTM_PATH, "w") as rttm:
        annotation.write_rttm(rttm)

# this pulls some data out of the rttm
print("parsing...")
onset_list, duration_list, speaker_list = diarization.parse_rttm(RTTM_PATH)

# this creates the timestamps we need to split the audio at to fit in whisper's 25mb limit
print("getting timestamps to split at...")
timestamp_list = processing.get_split_timestamps(onset_list, duration_list)

# pre-process the audio w.r.t. the timestamps
print("resampling and splitting audio...")
segment_temp_dir, segment_temp_file_list = processing.segment_resample_audio(AUDIO_PATH, timestamp_list)

# this is fluff but it's harmless fluff
print("processing filenames...")
segment_file_list = whisper.combine_segment_filenames(segment_temp_dir, segment_temp_file_list)

# create the transcripts
print("running segments through whisper...")
temp_transcript_list = whisper.process_all_segments(segment_file_list)

#combine them
print("combining transcripts...")
full_transcription_list = whisper.combine_transcripts(timestamp_list, temp_transcript_list)
with open("full_transcription.json", 'w') as file:
    json.dump(full_transcription_list, file, indent=2)

# combine the diarization
combined_diarization_list = diarization.combine_authors(onset_list, duration_list, speaker_list)

# match the words up with the speakers to build the final transcript
print("assembling final transcript...")
transcript = final_transcript(full_transcription_list, combined_diarization_list)
for turn in transcript:
    print(turn)
