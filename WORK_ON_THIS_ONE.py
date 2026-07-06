from diarization import Segment

def final_transcript(full_transcript, full_diarization):
    # step 1: label every word with the speaker who overlaps it the most
    labeled_words = assign_speakers(full_transcript, full_diarization)
    # step 2: clean up tiny speaker "blips" that are almost certainly diarization noise
    labeled_words = smooth_speakers(labeled_words)

    # step 3: group consecutive same-speaker words into (speaker, text) turns
    transcript_list = []
    current_speaker = None
    current_words = []
    for speaker, word in labeled_words:
        if speaker != current_speaker and current_words:
            transcript_list.append((current_speaker, ' '.join(current_words)))
            current_words = []
        current_speaker = speaker
        current_words.append(word['word'])
    if current_words:
        transcript_list.append((current_speaker, ' '.join(current_words)))
    return transcript_list

def assign_speakers(full_transcript, full_diarization):
    # for each word, add up how much time each speaker's segments cover it,
    # then hand the word to whoever covers the most. this beats only checking
    # where the word *ends*, because a word can end inside a tiny blip from
    # the wrong speaker while most of it sits in the right speaker's segment
    labeled_words = []
    base = 0
    previous_speaker = None

    for word in full_transcript:
        word_start, word_end = word['start'], word['end']

        # both lists are sorted by time, so any segment that ended before this
        # word started can never matter again -- skip past those for good
        while base < len(full_diarization) and full_diarization[base].end <= word_start:
            base += 1

        overlap_by_speaker = {}
        i = base
        while i < len(full_diarization) and full_diarization[i].start < word_end:
            segment = full_diarization[i]
            overlap = min(word_end, segment.end) - max(word_start, segment.start)
            if overlap > 0:
                overlap_by_speaker[segment.speaker] = overlap_by_speaker.get(segment.speaker, 0.0) + overlap
            i += 1

        if overlap_by_speaker:
            speaker = max(overlap_by_speaker, key=overlap_by_speaker.get)
        elif previous_speaker is not None:
            # the word fell in a silence gap between segments; whoever was
            # talking last is the safest guess
            speaker = previous_speaker
        elif base < len(full_diarization):
            # gap words before anyone has spoken get the first upcoming speaker
            speaker = full_diarization[base].speaker
        else:
            speaker = None

        labeled_words.append((speaker, word))
        previous_speaker = speaker

    return labeled_words

def smooth_speakers(labeled_words, max_blip_words=2, max_blip_seconds=0.8):
    if not labeled_words:
        return labeled_words

    # split the words into runs: consecutive words with the same speaker
    runs = []
    for speaker, word in labeled_words:
        if runs and runs[-1][0] == speaker:
            runs[-1][1].append(word)
        else:
            runs.append([speaker, [word]])

    # a one-or-two word run wedged between two runs of the same *other* speaker
    # is usually the diarizer flickering, not a real interjection -- relabel it.
    # relabeling can make neighbors mergeable (A blip A -> AAA), which can expose
    # new blips, so we keep sweeping until a full pass changes nothing
    changed = True
    while changed:
        changed = False

        # merge adjacent runs that now share a speaker
        merged_runs = [runs[0]]
        for run in runs[1:]:
            if run[0] == merged_runs[-1][0]:
                merged_runs[-1][1].extend(run[1])
            else:
                merged_runs.append(run)
        runs = merged_runs

        for i in range(1, len(runs) - 1):
            speaker, words = runs[i]
            duration = words[-1]['end'] - words[0]['start']
            if (runs[i - 1][0] == runs[i + 1][0]
                    and len(words) <= max_blip_words
                    and duration <= max_blip_seconds):
                runs[i][0] = runs[i - 1][0]
                changed = True

    # flatten the runs back out into (speaker, word) pairs
    return [(speaker, word) for speaker, words in runs for word in words]
