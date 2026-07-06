from enum import IntEnum
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

class Rttm(IntEnum):
    TYPE = 0
    FILE_ID = 1
    CHANNEL = 2
    ONSET = 3
    DURATION = 4
    SPEAKER = 7

@dataclass
class Segment:
    start: float
    end: float
    speaker: int

def diarize_audio(audio_path):
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook
    import torch

    os.environ["HF_HOME"] = "./pyannote_weights"
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=os.getenv("HF_PYANNOTE_TOKEN")
    )
    pipeline.to(torch.device("cuda"))
    file = pipeline.prepare_one(audio_path)
    with ProgressHook() as hook:
        output = pipeline.apply(file, hook = hook)
    return output.speaker_diarization

def parse_rttm(rttm_path):
    with open(rttm_path) as file:
        onset_list = []
        duration_list = []
        speaker_list = []
        for line in file:
            # split the line so that we can do stuff with it
            split_line = line.split()
            onset_list.append(float(split_line[Rttm.ONSET]))
            duration_list.append(float(split_line[Rttm.DURATION]))
            speaker_list.append(split_line[Rttm.SPEAKER])
    return onset_list, duration_list, speaker_list

def combine_authors(onset_list, duration_list, speaker_list, min_duration=0.04):
    combined_list = []
    current_speaker = None
    start = 0.0
    end = 0.0

    for onset, duration, speaker in zip(onset_list, duration_list, speaker_list):
        # too short to be real speech, so we nix it before it can
        # interrupt (and split up) the segment being built
        if duration < min_duration:
            continue
        speaker_id = int(speaker[-1:])
        # if the speaker is the same then we overwrite
        if speaker_id == current_speaker:
            end = onset + duration
        else:
            if current_speaker is not None:
                combined_list.append(Segment(start, end, current_speaker))
            start = onset
            end = onset + duration
            current_speaker = speaker_id
    if current_speaker is not None:
        combined_list.append(Segment(start, end, current_speaker))
    return combined_list