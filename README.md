# 命令记录

```
pip install deepspeed>=0.9.0

git clone https://github.com/microsoft/DeepSpeedExamples.git

cd DeepSpeedExamples/applications/DeepSpeed-Chat/

pip install -r requirements.txt

python train.py --actor-model facebook/opt-1.3b --reward-model facebook/opt-350m --deployment-type single_gpu

watch -n 1 nvidia-smi
```


