import verifiers as vf

"""
# install
vf-install haiku

# quick eval
vf-eval haiku (-m model_name in endpoints.py)

inference:
CUDA_VISIBLE_DEVICES=0 vf-vllm --model Qwen/Qwen3-4B-Thinking-2507 --enforce-eager --disable-log-requests --max-model-len 8192

training:
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 --config-file configs/zero3.yaml examples/train_haiku.py
"""

model_name = "Qwen/Qwen3-4B-Thinking-2507"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = vf.load_environment(env_id="haiku")

args = vf.grpo_defaults(run_name="haiku")
args.per_device_train_batch_size = 16
args.num_generations = 16
args.gradient_accumulation_steps = 8
args.max_steps = 200
args.max_tokens = 8192

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    peft_config=vf.lora_defaults(),
    args=args,
)
trainer.train()
