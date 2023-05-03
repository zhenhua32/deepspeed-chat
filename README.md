# 命令记录

```
pip install deepspeed>=0.9.0

git clone https://github.com/microsoft/DeepSpeedExamples.git

cd DeepSpeedExamples/applications/DeepSpeed-Chat/

pip install -r requirements.txt

python train.py --actor-model facebook/opt-1.3b --reward-model facebook/opt-350m --deployment-type single_gpu

watch -n 1 nvidia-smi

Launch command: bash /workspaces/deepspeed_chat/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/training_scripts/single_gpu/run_1.3b.sh /workspaces/deepspeed_chat/DeepSpeedExamples/applications/DeepSpeed-Chat/output/actor-models/1.3b /workspaces/deepspeed_chat/DeepSpeedExamples/applications/DeepSpeed-Chat/output/reward-models/350m '' '' /workspaces/deepspeed_chat/DeepSpeedExamples/applications/DeepSpeed-Chat/output/step3-models/1.3b

python chat.py --path  ${PATH-to-your-actor-model}
```

# 流程

总共有三个流程. 每个步骤都是要进去到目录下执行的.

每个步骤都在 `DeepSpeedExamples/applications/DeepSpeed-Chat/training/` 目录下.

## 1. 监督微调 SFT

TODO: 第一步就和文档上的示例差的好远.

目录是 `DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning`.

```bash
cd /workspaces/deepspeed_chat/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning
# 训练
bash training_scripts/single_gpu/run_1.3b.sh /workspaces/deepspeed_chat/output/actor-models/1.3b
# 评估
bash evaluation_scripts/run_prompt.sh
```

数据集有这些, 默认的脚本中只用了第一个 `Dahoas/rm-static`. 
TODO: 文档中说要对齐提示词, 需要看看有没有对比剩余的数据集.
(e.g., Dohas/rm-static uses "Human:" for queries and "Assistant:" for answers). 

```
Dahoas/rm-static
Dahoas/full-hh-rlhf
Dahoas/synthetic-instruct-gptj-pairwise
yitingxie/rlhf-reward-datasets
openai/webgpt_comparisons stanfordnlp/SHP
```

## 2. 微调奖励模型

目录是 `DeepSpeedExamples/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning`

```bash
cd /workspaces/deepspeed_chat/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning
# 训练
bash training_scripts/single_gpu/run_350m.sh /workspaces/deepspeed_chat/output/reward-models/350m
```

## 3. RLHF微调

目录是 `DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning`

```bash
cd /workspaces/deepspeed_chat/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning
# 训练
bash training_scripts/single_gpu/run_1.3b.sh /workspaces/deepspeed_chat/output/actor-models/1.3b /workspaces/deepspeed_chat/output/reward-models/350m '' '' /workspaces/deepspeed_chat/output/step3-models/1.3b
```

