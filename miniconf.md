# NTU-ML-MiniConf 2025





在[1]中以不同的 EfficientNet (B0,B1,B2,B3,B4)比較，並選定EfficientNetB4。
後續，EfficientNet-B4 在論文中達到 83.1% Top-1 精度，明顯優於同資料集上的 ResNet50。
- ResNet 家族主要以深度增加為主；EfficientNet 則是三維一起調：深、寬、解析度

在[2]中選用 ConvNeXt-Tiny（ConvNeXt-T）作為 backbone。加入 ACMix(x2) + Stacked-FFN Head (ACMix 就是 Hybrid CNN + Attention)。達到最佳。
- 模型 A：ConvNeXt-Tiny（without ACMix） → 驗證率約 69.6%
- 模型 B：ConvNeXt-Tiny + ACMix → 驗證率約 83.1%
- ConvNeXt 是一個 純 CNN（Non-Transformer） 的家族，類似 ViT 的架構風格，但保留卷積的效率。


### 根據以上總結可測試模型

| Model    | Type | Note |
| -------- | -------- | -------- |
| ResNet50    | Baseline     | Text     |
| EfficientNetB4   | _     | 因「compound scaling」在細粒度分類效果很好。     |
| ConvNeXt-T+ACMix   | _     | Hybrid CNN-Transformer     |
| ViT-B/16   | _     | 最基礎 Vision Transformer     |
| Swin Transformer (Tiny / Small)   | _     | 具層級結構，對 Herbal 影像佳     |
| DINOv2-Base   | _    | Foundation Model 是 Self-supervised SOTA，適合小dataset      |

###  Performance Matric
- 多類別分類（80–100 類）
- 細粒度分類（類別非常相似）
- 圖像分類
- 需要比較不同模型（CNN / ConvNeXt / ViT / Swin / DINOv2）
- [1] 有用 Top-1 和 Top-5 Accuracy, [2] 是 Accuracy

### 類別級別指標
- **Precision**
- **Recall**
- **F1-score**

$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$

### 準確率
- **Top-1 Accuracy**

$\text{Top-1 Acc} = \frac{\text{Number of correct predictions}}{\text{Total samples}}$

- **Top-5 Accuracy** 
    - 能反映模型是否理解相似類的語意

### 錯誤分析指標
- **Confusion Matrix** (錯誤分析指標)
了解哪些 herb 最容易被混淆（例如：川芎 vs 當歸）和 模型錯誤模式


Reference

[1] A novel Chinese herbal medicine classification approach based on EfficientNet
https://www.tandfonline.com/doi/full/10.1080/21642583.2021.1901159#d1e182

[2] Image recognition of traditional Chinese medicine based on deep learning
https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2023.1199803/full
