from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 1. 加载模型和tokenizer
model_path = "/Users/kkkk/Downloads/大语言模型-已下载/自己训练的/认知模型-四分类"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# 2. 定义分类标签（根据你的训练数据修改！）
labels = ["类别1", "类别2", "类别3", "类别4"]  # 替换为你的实际类别名称

# 3. 分类函数
def classify(text):
    # 文本编码
    inputs = tokenizer(text, 
                      padding=True, 
                      truncation=True, 
                      max_length=512, 
                      return_tensors="pt")
    
    # 预测
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取结果
    probs = torch.softmax(outputs.logits, dim=1)[0]
    predicted_id = torch.argmax(probs).item()
    
    return {
        "预测结果": labels[predicted_id],
        "类别概率": {labels[i]: f"{probs[i].item():.4f}" for i in range(len(labels))}
    }

# 4. 交互式分类
print("===== 文本分类器 =====")
print("输入 'exit' 退出程序\n")

while True:
    text = input("请输入要分类的文本: ")
    if text.lower() == 'exit':
        break
    
    result = classify(text)
    print(f"\n分类结果: {result['预测结果']}")
    print("概率分布:")
    for label, prob in result['类别概率'].items():
        print(f"  {label}: {prob}")
    print("-" * 50 + "\n")