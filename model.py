import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from esm import pretrained
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import esm
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report
from peft import get_peft_model, LoraConfig, TaskType
from collections import Counter
import random
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F

from collections import Counter

def read(file_paths, labels):
    samples = []
    seen_sequences = set()  # 用于记录已经处理过的截断序列
    max_length = 1024
    duplicates = []  # 用于存储重复的序列

    for file_path, label in zip(file_paths, labels):
        with open(file_path, "r") as f:
            sequence_lines = []
            for line in f:
                if line.startswith(">"):
                    if sequence_lines:
                        sequence = "".join(sequence_lines)
                        truncated_sequence = sequence[:max_length]
                        
                        # 检查该截断序列是否已经出现过
                        if truncated_sequence in seen_sequences:
                            duplicates.append(truncated_sequence)  # 存储重复的序列
                        else:
                            seen_sequences.add(truncated_sequence)  # 新序列，加入集合
                            samples.append((truncated_sequence, label))  # 只有第一次出现的序列才加入 samples
                        
                        sequence_lines = []
                else:
                    sequence_lines.append(line.strip())
            if sequence_lines:
                sequence = "".join(sequence_lines)
                truncated_sequence = sequence[:max_length]
                
                # 检查该截断序列是否已经出现过
                if truncated_sequence in seen_sequences:
                    duplicates.append(truncated_sequence)  # 存储重复的序列
                else:
                    seen_sequences.add(truncated_sequence)  # 新序列，加入集合
                    samples.append((truncated_sequence, label))  # 只有第一次出现的序列才加入 samples

    # 使用 Counter 统计重复的序列数量
    duplicate_count = Counter(duplicates)
    
    # 计算总的重复序列数
    total_duplicate_count = sum(duplicate_count.values())

    # # 打印重复的序列数量
    # if duplicate_count:
    #     print(f"重复的序列数量：{total_duplicate_count} 条")
    #     for sequence, count in duplicate_count.items():
    #         print(f"序列：{sequence}, 出现次数：{count}")
    # else:
    #     print("没有重复的序列")
        
    return samples


class ContrastiveCas12Dataset(Dataset):
    def __init__(self, samples, batch_converter, num_classes, num_positive_samples=1, num_negative_samples=2, max_length=1080):

        self.samples = samples
        self.batch_converter = batch_converter
        self.num_classes = num_classes
        self.num_positive_samples = num_positive_samples
        self.num_negative_samples = num_negative_samples
        self.max_length = max_length
        self.label_to_samples = {i: [] for i in range(num_classes)}

        # 将样本按类别分类
        for sequence, label in self.samples:
            self.label_to_samples[label].append((sequence, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sequence, label = self.samples[idx]
        
        # 构建 anchor，正样本和负样本
        anchor = sequence
        positive_samples = self._get_positive_samples(label)
        negative_samples = self._get_negative_samples(label)

        # 数据转换
        valid_chars = set(self.batch_converter.alphabet.standard_toks)
        anchor = ''.join([char if char in valid_chars else "X" for char in anchor])[:self.max_length]
        positive_samples = [
            ''.join([char if char in valid_chars else "X" for char in ps])[:self.max_length] 
            for ps in positive_samples
        ]
        negative_samples = [
            ''.join([char if char in valid_chars else "X" for char in ns])[:self.max_length] 
            for ns in negative_samples
        ]
        
        # 转换为模型输入
        data_anchor = [(f"anchor_{idx}", anchor)]
        data_positives = [(f"positive_{idx}_{i}", ps) for i, ps in enumerate(positive_samples)]
        data_negatives = [(f"negative_{idx}_{i}", ns) for i, ns in enumerate(negative_samples)]

        _, _, batch_tokens_anchor = self.batch_converter(data_anchor)
        batch_tokens_positives = [self.batch_converter([dp])[2].squeeze(0) for dp in data_positives]
        batch_tokens_negatives = [self.batch_converter([dn])[2].squeeze(0) for dn in data_negatives]

        return {
            "label": torch.tensor(label, dtype=torch.long),
            "anchor": batch_tokens_anchor.squeeze(0),
            "positive": batch_tokens_positives,  # 返回正样本列表
            "negative": batch_tokens_negatives   # 返回负样本列表
        }

    def _get_positive_samples(self, label):
        """ 获取多个同类别的正样本 """
        positive_samples = self.label_to_samples[label]
        return random.choices([ps[0] for ps in positive_samples], k=self.num_positive_samples)

    def _get_negative_samples(self, label):
        """ 获取多个不同类别的负样本 """
        negative_samples = []
        for _ in range(self.num_negative_samples):
            negative_label = random.choice([l for l in range(self.num_classes) if l != label])
            # print(negative_label)
            # print(self.label_to_samples[negative_label])
            negative_samples.append(random.choice(self.label_to_samples[negative_label])[0])
        return negative_samples



def collate_fn(batch):
    max_len = 1024  # 设置最大长度
    anchors = torch.stack([
        torch.cat([item["anchor"], torch.zeros(max(0, max_len - item["anchor"].shape[0]), dtype=torch.long)])[:max_len]
        for item in batch
    ])
    positives = torch.stack([
        torch.cat([item["positive"][i], torch.zeros(max(0, max_len - item["positive"][i].shape[0]), dtype=torch.long)])[:max_len]
        for item in batch for i in range(len(item["positive"]))  # 遍历每个正样本
    ])
    negatives = torch.stack([
        torch.cat([item["negative"][i], torch.zeros(max(0, max_len - item["negative"][i].shape[0]), dtype=torch.long)])[:max_len]
        for item in batch for i in range(len(item["negative"]))  # 遍历每个负样本
    ])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {
        "label": labels,
        "anchor": anchors,
        "positive": positives,
        "negative": negatives
    }
class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get(self, key, default=None):
        return getattr(self, key, default)

# 包装 ESM 模型，添加 config 属性
class ESMWithConfig(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.config = Config(
            use_return_dict=False,
            hidden_size=base_model.embed_dim,
            num_labels=14,
            tie_word_embeddings=False
        )
        self.embed_dim = base_model.embed_dim  # 暴露嵌入维度

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

class EsmClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, hidden_dropout_prob=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features):
        x = features  # 取 <s> token (相当于 [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
class CasClassifier(nn.Module):
    def __init__(self, model, num_classes, lora_config, hidden_dropout_prob=0.1):
        super(CasClassifier, self).__init__()
        self.encoder = get_peft_model(model, lora_config)
        
        hidden_size = self.encoder.base_model.embed_dim
        self.classifier = EsmClassificationHead(hidden_size, num_classes, hidden_dropout_prob)

    def forward(self, tokens, attention_mask=None):
        outputs = self.encoder.base_model(tokens,repr_layers=[33])
        pooled_output = outputs["representations"][33][:,0,:]
        # print(pooled_output.shape)
        logits = self.classifier(pooled_output)
        # print(logits.shape)
        return logits,pooled_output



# 训练函数
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, criterion,writer=None, epoch=0, temperature=0.07):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    global_step = epoch * len(dataloader)

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch}")):

        labels = batch["label"].to(device)
        anchors = batch["anchor"].to(device)
        positives = batch["positive"].to(device)
        negatives = batch["negative"].to(device)
        optimizer.zero_grad()
        logits_anchor, anchor_embeddings = model(anchors)
        logits_positive, positive_embeddings = model(positives)
        logits_negative, negative_embeddings = model(negatives)
        ce_loss = criterion(logits_anchor, labels)
        contrastive_loss_value = 0
        for i in range(len(anchors)):
            anchor_emb = anchor_embeddings[i]
            pos_embs = positive_embeddings[i]
            neg_embs = negative_embeddings[i]
            contrastive_loss_value += contrastive_loss(anchor_emb, pos_embs, neg_embs, temperature)
        contrastive_loss_value /= len(anchors)
        total_loss_value = ce_loss + 0.1*contrastive_loss_value
        total_loss_value.backward()
        optimizer.step()
        total_loss += total_loss_value.item()
        _, predicted = torch.max(logits_anchor, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        if writer and (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
            writer.add_scalar("Train/Loss_Batch", avg_loss, global_step + batch_idx)
            writer.add_scalar("Train/Accuracy", report["accuracy"], global_step + batch_idx)
            writer.add_scalar("Train/Precision", report["macro avg"]["precision"], global_step + batch_idx)
            writer.add_scalar("Train/Recall", report["macro avg"]["recall"], global_step + batch_idx)
            writer.add_scalar("Train/F1", report["macro avg"]["f1-score"], global_step + batch_idx)
            for cls, metrics in report.items():
                try:
                    cls_int = int(cls) 
                    writer.add_scalar(f"Train/Class_{cls_int}_Precision", metrics["precision"], global_step + batch_idx)
                    writer.add_scalar(f"Train/Class_{cls_int}_Recall", metrics["recall"], global_step + batch_idx)
                    writer.add_scalar(f"Train/Class_{cls_int}_F1", metrics["f1-score"], global_step + batch_idx)
                except ValueError:
                    pass
        global_step += 1

    epoch_loss = total_loss / len(dataloader)
    epoch_accuracy = accuracy_score(all_labels, all_preds)
    return epoch_loss


def evaluate_epoch(model, dataloader, criterion, writer=None, epoch=0, temperature=0.07):
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    all_preds = []
    all_labels = []
    global_step = epoch * len(dataloader)

    # 禁用梯度计算，以节省内存和加速推理
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating Epoch {epoch}")):
            # labels = batch["label"]
            # anchors = batch["anchor"]
            # positives = batch["positive"]
            # negatives = batch["negative"]

            labels = batch["label"].to(device)
            anchors = batch["anchor"].to(device)
            positives = batch["positive"].to(device)
            negatives = batch["negative"].to(device)

            logits_anchor, anchor_embeddings = model(anchors)
            logits_positive, positive_embeddings = model(positives)
            logits_negative, negative_embeddings = model(negatives)
            ce_loss = criterion(logits_anchor, labels)
            contrastive_loss_value = 0
            for i in range(len(anchors)):
                anchor_emb = anchor_embeddings[i]
                pos_embs = positive_embeddings[i]
                neg_embs = negative_embeddings[i] 
                contrastive_loss_value += contrastive_loss(anchor_emb, pos_embs, neg_embs, temperature)
            contrastive_loss_value /= len(anchors)
            total_loss_value = ce_loss + 0.1*contrastive_loss_value 
            total_loss += total_loss_value.item()
            _, predicted = torch.max(logits_anchor, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = accuracy_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        if writer:
            writer.add_scalar("Eval/Loss", epoch_loss, global_step)
            writer.add_scalar("Eval/Accuracy", epoch_accuracy, global_step)
            writer.add_scalar("Eval/Precision", report["macro avg"]["precision"], global_step)
            writer.add_scalar("Eval/Recall", report["macro avg"]["recall"], global_step)
            writer.add_scalar("Eval/F1", report["macro avg"]["f1-score"], global_step)
            for cls, metrics in report.items():
                try:
                    cls_int = int(cls)
                    writer.add_scalar(f"Eval/Class_{cls_int}_Precision", metrics["precision"], global_step)
                    writer.add_scalar(f"Eval/Class_{cls_int}_Recall", metrics["recall"], global_step)
                    writer.add_scalar(f"Eval/Class_{cls_int}_F1", metrics["f1-score"], global_step)
                except ValueError:
                    pass

    return epoch_loss, epoch_accuracy

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2, num_classes=13, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha) 
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        target_probs = probs.gather(1, targets.view(-1, 1))
        device = targets.device 
        self.alpha = self.alpha.to(device) 
        alpha_weights = self.alpha[targets.view(-1)]
        focal_loss = -alpha_weights * (1 - target_probs) ** self.gamma *       torch.log(target_probs)
        if self.reduction == 'none':
            return focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
def main():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    new_dir = '/home/zhengheyuan/lyy_cas'
    os.environ['TORCH_HOME'] = new_dir
    trained_model_dir = os.path.join(new_dir, "trained_model_cas_all_CE_0.1constractive_more_negative")
    os.makedirs(trained_model_dir, exist_ok=True)
    log_dir = os.path.join(new_dir, "logs_cas_all_CE_0.1constractive_more_negative")
    if dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None
    base_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()   
    base_model = ESMWithConfig(base_model)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj",  "fc1", "fc2" ]
    )
    for param in base_model.parameters():
        param.requires_grad = False

    num_classes = 13
    hidden_dropout_prob = 0.1
    classifier_model = CasClassifier(base_model, num_classes, lora_config, hidden_dropout_prob).to(device)
    for param in classifier_model.classifier.parameters():
        param.requires_grad = True
    batch_converter = alphabet.get_batch_converter()
    num_epochs = 30
    batch_size = 1
    learning_rate = 2e-5
    data_dir = "/home/zhengheyuan/lyy_cas/data/cas_all_filter"
    faa_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    print(faa_files)
    labels = list(range(len(faa_files)))
    file_labels = dict(zip(faa_files, labels))
    for file, label in file_labels.items():
        print(f"File: {file}, Label: {label}")
    data = read(faa_files,labels)
    print(len(data))
    labels = [item[1] for item in data]
    data_train, data_test,_,_ = train_test_split(data,labels, test_size=0.2, random_state=42, stratify=labels)
    print(len(data_train))
    print(len(data_test))

    train_dataset = ContrastiveCas12Dataset(data_train, batch_converter, num_classes=13) 
    test_dataset = ContrastiveCas12Dataset(data_test, batch_converter, num_classes=13)
      

    train_sampler = DistributedSampler(train_dataset,shuffle=True)
    test_sampler = DistributedSampler(test_dataset, shuffle=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    classifier_model = torch.nn.parallel.DistributedDataParallel(
        classifier_model,
        device_ids=[int(os.environ["LOCAL_RANK"])],
        find_unused_parameters=True,
    )

    optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss() 


    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_sampler.set_epoch(epoch)
        # 训练阶段
        train_loss = train_epoch(classifier_model, train_loader, optimizer, criterion, writer=writer, epoch=epoch)
        if writer:
            writer.add_scalar("Train/Loss_Epoch", train_loss, epoch)
        # 测试阶段
        test_loss, test_accuracy = evaluate_epoch(classifier_model, test_loader, criterion, writer=writer, epoch=epoch)
        # if writer:
        #     writer.add_scalar("Test/Loss_Epoch", test_loss, epoch)
        #     writer.add_scalar("Test/Accuracy_Epoch", test_accuracy, epoch)
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        # 保存模型
        if dist.get_rank() == 0:  # 确保只保存一次
            # 每个 epoch 保存模型到 trained_model 文件夹中
            model_save_path = os.path.join(trained_model_dir, f"cas_all_filter_model_epoch_{epoch + 1}.pt")
            torch.save(classifier_model.module.state_dict(), model_save_path)
            print(f"模型已保存到 {model_save_path}")                
    if writer:
        writer.close()
    dist.destroy_process_group()
if __name__ == "__main__":
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 张显卡进行训练")
    main()
