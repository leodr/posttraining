from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model_id = "LiquidAI/LFM2.5-1.2B-Base"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype="bfloat16",
    #   attn_implementation="flash_attention_2" <- uncomment on compatible GPU
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

prompt = """The following is a conversation between a user and a model. The model is a helpful assistant.

Assistant: Hello! How can I help you today?
User: What is the weather in Tokyo?"""

input_ids = tokenizer(
    prompt,
    return_tensors="pt",
).to(model.device)

output = model.generate(
    **input_ids,
    do_sample=True,
    temperature=0.3,
    min_p=0.15,
    repetition_penalty=1.05,
    max_new_tokens=512,
    streamer=streamer,
)
