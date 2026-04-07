# hpml_multimodal_nanochat
7:47 PMPretraining a small GPT (nanochat) on COCO Captions, then extending it into a multimodal vision-language model with ViT/CLIP encoder and cross-attention fusion for image captioning. Profiling data loading, training, and inference, identify bottlenecks and apply optimizations (AMP, torch.compile, FlashAttention, KV-cache) with before/after. 
