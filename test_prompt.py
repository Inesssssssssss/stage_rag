from faster_whisper import WhisperModel
import time

model_size = "large-v3"

# Run on GPU with FP16
#model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

# or run on CPU with INT8
#model = WhisperModel(model_size, device="cpu", compute_type="int8")
start = time.time()
segments, info = model.transcribe("llm_planner_vlm_llm/audio/No.m4a", beam_size=5, suppress_tokens=[-1, 11, 13])
end1 = time.time()
transcribe_t = end1 - start

transcript = ""
for segment in segments:
    # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    transcript += segment.text
    print("seg,ents tokens : ", segment.tokens)

end2 = time.time()
total_t = end2 - start

print("Transcription of audio took : %.2fs" % (total_t))
print("Transcript :", transcript.lstrip(' '))
