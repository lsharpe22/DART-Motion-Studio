# Evaluation and comparison with original DART paper

## Protocol

- Paper citation/version and checkpoint name: [fill in]
- Original DART evaluation metric and hardware: [cite paper]
- Our GPU and software: A40; Python 3.8.20; PyTorch 2.1.0+cu118 (verify each run)
- Seed: 0; guidance: 1.0; batch size: 1; predicted joints: enabled; repetitions: 5
- Run at least 3 trials per prompt; distinguish first/cold from warmed-up requests.
- Record timestamp, prompt, repetitions, frames, wall-clock seconds, frames/second, and visual observations.

| Prompt | Repetitions | Frames | First/warm | Seconds | FPS (frames/seconds) | Notes |
|---|---:|---:|---|---:|---:|---|
| walk forward | 5 | [run] | cold | [run] | [calc] | [observations] |
| walk forward | 5 | [run] | warm | [run] | [calc] | [observations] |
| turn left (continuation) | 5 | [run] | warm | [run] | [calc] | [observations] |

## Past observations (not controlled replication)

- Migrated A40 persistent worker initialization: **8.27 s** (single observation).
- Earlier first API generation: **13.34 s**, 42 frames.
- Earlier continuation: **0.74 s**, 40 new frames.
- These are previous observations, not 3+ repeated A40 trials under identical settings.

## Paper comparison to complete

| Dimension | DART paper (cite figure/table) | Our deployment | Direct comparison? |
|---|---|---|---|
| GPU | [paper] | NVIDIA A40 | No if GPUs differ |
| Model/checkpoint | [paper] | checkpoint_300000.pt | Verify |
| Inference mode | [paper] | Persistent worker + browser API | Usually different |
| Sampling setup | [paper] | Guidance 1.0, predicted joints | Verify |
| Throughput | [paper] | [controlled results] | Only if measurement definitions match |
| Quality | [paper metric] | Visual observations | Not quantitative head-to-head |

## Limitations

The browser renders 22-joint motion sequences and does not by itself reproduce the paper's full standardized evaluation suite. Some walking motion includes lateral wobble. Mixamo retargeting can look stiff; distinguish animation/rig artifacts from generated motion artifacts. Do not claim the pretrained DART model was retrained.
