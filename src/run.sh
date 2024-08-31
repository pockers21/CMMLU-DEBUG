rm record
rm -rf /root/CMMLU/results/*
#python qwen2.py --device cpu  --filter --loopcnt 1
python qwen2_eagle.py --device cpu  --filter --loopcnt 1
python qwen2_eagle.py --device auto  --filter --loopcnt 1
