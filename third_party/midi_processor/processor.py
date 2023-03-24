import pretty_midi


RANGE_NOTE_ON = 128
RANGE_NOTE_OFF = 128
RANGE_VEL = 32
RANGE_TIME_SHIFT = 100
RANGE_DURATION = 100

START_IDX = {
    'note_on': 0,
    'note_off': RANGE_NOTE_ON,
    'time_shift': RANGE_NOTE_ON + RANGE_NOTE_OFF,
    'velocity': RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT
}


class SustainAdapter:
    def __init__(self, time, type):
        self.start =  time
        self.type = type


class SustainDownManager:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.managed_notes = []
        self._note_dict = {} # key: pitch, value: note.start

    def add_managed_note(self, note: pretty_midi.Note):
        self.managed_notes.append(note)

    def transposition_notes(self):
        for note in reversed(self.managed_notes):
            try:
                note.end = self._note_dict[note.pitch]
            except KeyError:
                note.end = max(self.end, note.end)
            self._note_dict[note.pitch] = note.start


# Divided note by note_on, note_off
class SplitNote:
    def __init__(self, type, time, value, velocity):
        ## type: note_on, note_off
        self.type = type
        self.time = time
        self.velocity = velocity
        self.value = value

    def __repr__(self):
        return '<[SNote] time: {} type: {}, value: {}, velocity: {}>'\
            .format(self.time, self.type, self.value, self.velocity)


class Event:
    def __init__(self, event_type, value):
        self.type = event_type
        self.value = value

    def __repr__(self):
        return '<Event type: {}, value: {}>'.format(self.type, self.value)

    def to_int(self):
        return START_IDX[self.type] + self.value

    @staticmethod
    def from_int(int_value):
        info = Event._type_check(int_value)
        return Event(info['type'], info['value'])

    @staticmethod
    def _type_check(int_value):
        range_note_on = range(0, RANGE_NOTE_ON)
        range_note_off = range(RANGE_NOTE_ON, RANGE_NOTE_ON+RANGE_NOTE_OFF)
        range_time_shift = range(RANGE_NOTE_ON+RANGE_NOTE_OFF,RANGE_NOTE_ON+RANGE_NOTE_OFF+RANGE_TIME_SHIFT)

        valid_value = int_value

        if int_value in range_note_on:
            return {'type': 'note_on', 'value': valid_value}
        elif int_value in range_note_off:
            valid_value -= RANGE_NOTE_ON
            return {'type': 'note_off', 'value': valid_value}
        elif int_value in range_time_shift:
            valid_value -= (RANGE_NOTE_ON + RANGE_NOTE_OFF)
            return {'type': 'time_shift', 'value': valid_value}
        else:
            valid_value -= (RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT)
            return {'type': 'velocity', 'value': valid_value}


def _divide_note(notes):
    result_array = []
    notes.sort(key=lambda x: x.start)

    for note in notes:
        on = SplitNote('note_on', note.start, note.pitch, note.velocity)
        off = SplitNote('note_off', note.end, note.pitch, None)
        result_array += [on, off]
    return result_array


def _merge_note(snote_sequence):
    note_on_dict = {}
    result_array = []

    for snote in snote_sequence:
        # print(note_on_dict)
        if snote.type == 'note_on':
            note_on_dict[snote.value] = snote
        elif snote.type == 'note_off':
            try:
                on = note_on_dict[snote.value]
                off = snote
                if off.time - on.time == 0:
                    continue
                result = pretty_midi.Note(on.velocity, snote.value, on.time, off.time)
                result_array.append(result)
            except:
                print('info removed pitch: {}'.format(snote.value))
    return result_array


def _snote2events(snote: SplitNote, prev_vel: int):
    result = []
    if snote.velocity is not None:
        modified_velocity = snote.velocity // 4
        if prev_vel != modified_velocity:
            result.append(Event(event_type='velocity', value=modified_velocity))
    result.append(Event(event_type=snote.type, value=snote.value))
    return result


def _event_seq2snote_seq(event_sequence):
    timeline = 0
    velocity = 0
    snote_seq = []

    for event in event_sequence:
        if event.type == 'time_shift':
            timeline += ((event.value+1) / 100)
        if event.type == 'velocity':
            velocity = event.value * 4
        else:
            snote = SplitNote(event.type, timeline, event.value, velocity)
            snote_seq.append(snote)
    return snote_seq


def _make_time_sift_events(prev_time, post_time):
    time_interval = int(round((post_time - prev_time) * 100))
    results = []
    while time_interval >= RANGE_TIME_SHIFT:
        results.append(Event(event_type='time_shift', value=RANGE_TIME_SHIFT-1))
        time_interval -= RANGE_TIME_SHIFT
    if time_interval == 0:
        return results
    else:
        return results + [Event(event_type='time_shift', value=time_interval-1)]


def _control_preprocess(ctrl_changes):
    sustains = []

    manager = None
    for ctrl in ctrl_changes:
        if ctrl.value >= 64 and manager is None:
            # sustain down
            manager = SustainDownManager(start=ctrl.time, end=None)
        elif ctrl.value < 64 and manager is not None:
            # sustain up
            manager.end = ctrl.time
            sustains.append(manager)
            manager = None
        elif ctrl.value < 64 and len(sustains) > 0:
            sustains[-1].end = ctrl.time
    return sustains


def _note_preprocess(susteins, notes):
    note_stream = []

    if susteins:    # if the midi file has sustain controls
        for sustain in susteins:
            for note_idx, note in enumerate(notes):
                if note.start < sustain.start:
                    note_stream.append(note)
                elif note.start > sustain.end:
                    notes = notes[note_idx:]
                    sustain.transposition_notes()
                    break
                else:
                    sustain.add_managed_note(note)

        for sustain in susteins:
            note_stream += sustain.managed_notes
    
    else:       # else, just push everything into note stream
        for note_idx, note in enumerate(notes):
            note_stream.append(note)

    note_stream.sort(key= lambda x: x.start)
    return note_stream


def encode_midi(file_path):     # encode_midi_modified
    events = []
    notes = []
    try:
        mid = pretty_midi.PrettyMIDI(midi_file=file_path)
    except OSError:
        print(f"{file_path} could not be opened. Invalid byte")
        return []
    except Exception as e:
        print(f"Excpetion opening {file_path}: {e}")
        return []

    for inst in mid.instruments:
        if inst.is_drum:
            continue
        inst_notes = inst.notes
        # ctrl.number is the number of sustain control. If you want to know abour the number type of control,
        # see https://www.midi.org/specifications-old/item/table-3-control-change-messages-data-bytes-2
        ctrls = _control_preprocess([ctrl for ctrl in inst.control_changes if ctrl.number == 64])
        notes += _note_preprocess(ctrls, inst_notes)

    notes.sort(key=lambda x: x.start)

    #  0  - 127 Note on
    # 128 - 227 Duration
    # 228 - 327 Timeshift
    # 328 - 359 Velocity
    try:
        cur_time, cur_vel = notes[0].start, notes[0].velocity // 4
    except IndexError:
        print(f"{file_path} has 0 notes")
        return []

    # Put in timeshifts to the start of the song
    if cur_time > 0:
        timeshift = round(cur_time * 100)
        while timeshift > 100:
            events.append(327)
            timeshift -= 100
        if timeshift > 0:
            events.append(timeshift + 228)

    # Put in first note
    events.append(min(cur_vel + 328, 359))
    events.append(notes[0].pitch)
    duration = max(1, round(notes[0].get_duration() * 100))
    while duration > 100:
        events.append(227)
        duration -= 100
    events.append(duration + 128)

    # Put in the rest if the notes
    for note in notes[1:]:
        # Put in timeshift events
        timeshift = round((note.start  - cur_time) * 100)
        while timeshift > 100:
            events.append(327)
            timeshift -= 100
        if timeshift > 0:
            events.append(timeshift + 228)
            cur_time = note.start

        # Put in velocity events
        velocity = note.velocity // 4
        if velocity != cur_vel:
            events.append(min(velocity + 328, 359))
            cur_vel = velocity

        # Put in note on event
        events.append(note.pitch)

        # Put in duration events (10ms resolution)
        duration = max(1, round(note.get_duration() * 100))
        while duration > 100:
            events.append(227)
            duration -= 100
        if duration > 0:
            events.append(duration + 128)


    return [int(e) for e in events]


def encode_midi_original(file_path):
    events = []
    notes = []
    try:
        mid = pretty_midi.PrettyMIDI(midi_file=file_path)
    except OSError:
        print(f"{file_path} could not be opened")
        return []
    except Exception as e:
        print(f"Excpetion opening {file_path}: {e}")
        return []

    for inst in mid.instruments:
        if inst.is_drum:
            continue
        inst_notes = inst.notes
        # ctrl.number is the number of sustain control. If you want to know abour the number type of control,
        # see https://www.midi.org/specifications-old/item/table-3-control-change-messages-data-bytes-2
        ctrls = _control_preprocess([ctrl for ctrl in inst.control_changes if ctrl.number == 64])
        notes += _note_preprocess(ctrls, inst_notes)

    dnotes = _divide_note(notes)

    # print(dnotes)
    dnotes.sort(key=lambda x: x.time)
    # print('sorted:')
    # print(dnotes)
    cur_time = 0
    cur_vel = 0
    for snote in dnotes:
        events += _make_time_sift_events(prev_time=cur_time, post_time=snote.time)
        events += _snote2events(snote=snote, prev_vel=cur_vel)
        # events += _make_time_sift_events(prev_time=cur_time, post_time=snote.time)

        cur_time = snote.time
        cur_vel = snote.velocity

    return [e.to_int() for e in events]


def decode_midi(idx_array, file_path=None):     # decode_midi_modified
    mid = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(1, False, "Modified")
    notes = []
    cur_time, cur_vel = 0, 0

    #  0  - 127 Note on
    # 128 - 227 Duration
    # 228 - 327 Timeshift
    # 328 - 359 Velocity
    for event in idx_array:
        if 0 <= event <= 127:
            notes.append(pretty_midi.Note(cur_vel, event, cur_time, cur_time))
        elif 128 <= event <= 227:
            notes[-1].end += (event-128) / 100
        elif 228 <= event <= 327:
            cur_time += (event-228) / 100
        elif 328 <= event <= 359:
            cur_vel = (event-328) * 4

    instrument.notes = notes
    mid.instruments.append(instrument)
    if file_path is not None:
        mid.write(file_path)
    return mid


def decode_midi_original(idx_array, file_path=None):
    event_sequence = [Event.from_int(idx) for idx in idx_array]
    # print(event_sequence)
    snote_seq = _event_seq2snote_seq(event_sequence)
    note_seq = _merge_note(snote_seq)
    note_seq.sort(key=lambda x:x.start)

    mid = pretty_midi.PrettyMIDI()
    # if want to change instument, see https://www.midi.org/specifications/item/gm-level-1-sound-set
    instrument = pretty_midi.Instrument(1, False, "Developed By Yang-Kichang")
    instrument.notes = note_seq

    mid.instruments.append(instrument)
    if file_path is not None:
        mid.write(file_path)
    return mid


if __name__ == '__main__':
    encoded = encode_midi('bin/ADIG04.mid')
    print(encoded)
    decided = decode_midi(encoded,file_path='bin/test.mid')

    ins = pretty_midi.PrettyMIDI('bin/ADIG04.mid')
    print(ins)
    print(ins.instruments[0])
    for i in ins.instruments:
        print(i.control_changes)
        print(i.notes)

