# Satellite-Land-Use-Classification-with-Deep-Learning-EuroSAT+ ResNet18-

This project demonstrates how **satellite imagery** can be leveraged for **land use / land cover classification** using deep learning.  
We apply **transfer learning with ResNet18** on the [EuroSAT dataset](https://github.com/phelber/EuroSAT), which contains Sentinel-2 satellite images across **10 classes**.

---

🚀 Project Overview
- **Goal:** Classify satellite images into land use categories such as Residential, River, Sea/Lake, Forest, etc.  
- **Approach:** Transfer learning with ResNet18 pre-trained on ImageNet.  
- **Dataset:** EuroSAT (RGB, 27,000+ labeled images, 10 classes).  
- **Results:** Achieved **~93% validation accuracy in just 3 epochs**.  

---

## 🛠 Tech Stack
- **Language:** Python 3.9   
- **Frameworks:** PyTorch, torchvision  
- **Tools:** Google Colab / Jupyter Notebook  
- **Visualization:** Matplotlib, Seaborn  

---

## 📂 Dataset
EuroSAT contains Sentinel-2 satellite images across the following **10 categories**:

1. AnnualCrop  
2. Forest  
3. HerbaceousVegetation  
4. Highway  
5. Industrial  
6. Pasture  
7. PermanentCrop  
8. Residential  
9. River  
10. SeaLake  

Each class has **~2,000–3,000 images** of size **64x64 pixels**.

---

## ⚙️ Training Setup
- **Model:** ResNet18 (transfer learning, pretrained on ImageNet)  
- **Optimizer:** Adam (lr=0.001)  
- **Loss:** CrossEntropyLoss  
- **Epochs:** 3 (demo run; higher can yield better results)  
- **Hardware:** Google Colab (GPU)  

---

## 📊 Results

##Sample Dataset
dataset_samples.png

## NDVI Example
ndvi_example.png

## Training Curves
training_curves.png

### Prediction Results
Ppredictions.png


📈 Final Accuracy: **93.24% (Validation)**

---

## ✨ Key Takeaways
- Transfer learning drastically reduces training time.  
- Even with a few epochs, we can achieve **state-of-the-art performance on small datasets**.  
- Such models can be extended to real-world **agriculture monitoring, land cover mapping, and environmental planning**.  


---

## 📜 References
- [EuroSAT Dataset](https://github.com/phelber/EuroSAT)  
- [ResNet Paper](https://arxiv.org/abs/1512.03385)  
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)  

---

## 👨‍💻 Author
- **Deepak Battula**


---
