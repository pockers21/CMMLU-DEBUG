export PYTHONPATH=/root/eagle-qwen2-test/eagle
 
#python  gen_baseline_answer_qwen2.py --base-model-path /root/autodl-fs/Qwen2-7B-Instruct \
#                                      --ea-model-path  /root/autodl-fs/yuhuili/EAGLE-Qwen2-7B-Instruct



#python  gen_ea_answer_qwen2.py  --base-model-path /root/autodl-fs/Qwen2-7B-Instruct \3
#                                  --ea-model-path  /root/autodl-fs/yuhuili/EAGLE-Qwen2-7B-Instruct


python  gen_ea_answer_qwen2_summary.py  \
                        --base-model-path /root/autodl-fs/Qwen2-7B-Instruct \
                        --ea-model-path  /root/autodl-fs/yuhuili/EAGLE-Qwen2-7B-Instruct

python  gen_baseline_answer_qwen2_summary.py  \
                        --base-model-path /root/autodl-fs/Qwen2-7B-Instruct \
                        --ea-model-path  /root/autodl-fs/yuhuili/EAGLE-Qwen2-7B-Instruct
