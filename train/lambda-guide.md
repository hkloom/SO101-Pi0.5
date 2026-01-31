# Lambda Cloud GH200 Setup Guide for LeRobot PI 0.5 Training (v2)

This guide uses a pre-converted v3.0 dataset transferred from local machine.

---

## Prerequisite - API Keys (needed for setup_lambda.sh)
- WandB
- HuggingFace

---

## Step 1: Launch Lambda Instance
1. Go to https://cloud.lambdalabs.com
2. Launch a H100 or B200 instance
3. Note the IP address
4. Ensure your SSH key is added to Lambda (Settings > SSH Keys)

---

## Step 2: SSH into Instance
```zsh
LAMBDA_IP=0.0.0.0
ssh ubuntu@$LAMBDA_IP
mkdir -p ~/lerobot-training/dataset
```

---

# Step 3: Send Scripts
```zsh
scp ./train/train.py ubuntu@$LAMBDA_IP:~/lerobot-training/
scp ./train/setup_lambda.sh ubuntu@$LAMBDA_IP:~/lerobot-training/
```

---

# Step 4: Run Setup Script
```zsh
cd ~/lerobot-training
chmod +x setup_lambda.sh
./setup_lambda.sh
hf auth whoami
```

---

# Step 5: Start Training
```zsh
tmux new -s lerobot
cd ~/lerobot-training && python train.py
```
---

## Optional - Download Trained Policy (it is also uploaded to HF)
```bash
scp -r ubuntu@$LAMBDA_IP:~/lerobot-training/trained-pi05 ./
```

---