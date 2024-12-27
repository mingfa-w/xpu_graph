# 1. RUN Llama

```
python test_llama.py --model_path ./Llama-2-7b-hf/ --device "cuda:0"
```
If you want to use fa pass in xpu-graph instead of sdpa, you can add '"_attn_implementation": "eager"' in config.json 

# 2. RUN Qwen 
```
python test_llama.py --model_path ./Qwen-7B-Chat/ --device "cuda:0"
```

