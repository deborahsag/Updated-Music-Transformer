import torch

from third_party.midi_processor.processor import RANGE_NOTE_ON, RANGE_NOTE_OFF, RANGE_VEL, RANGE_TIME_SHIFT, RANGE_DURATION

SEPARATOR               = "========================="

# Taken from the paper
ADAM_BETA_1             = 0.9
ADAM_BETA_2             = 0.98
ADAM_EPSILON            = 10e-9

LR_DEFAULT_START        = 1.0
SCHEDULER_WARMUP_STEPS  = 4000
# LABEL_SMOOTHING_E       = 0.1

# DROPOUT_P               = 0.1

TOKEN_END               = RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_VEL + RANGE_TIME_SHIFT
TOKEN_END_NEW_NOTATION  = RANGE_NOTE_ON + RANGE_DURATION + RANGE_TIME_SHIFT + RANGE_VEL
TOKEN_PAD               = TOKEN_END + 1
TOKEN_PAD_NEW_NOTATION  = TOKEN_END_NEW_NOTATION + 1

VOCAB_SIZE              = TOKEN_PAD + 1
VOCAB_SIZE_NEW_NOTATION = TOKEN_PAD_NEW_NOTATION + 1

TORCH_FLOAT             = torch.float32
TORCH_INT               = torch.int32

TORCH_LABEL_TYPE        = torch.long

PREPEND_ZEROS_WIDTH     = 4
