**##Skin Disease Classification Using Hybrid CNN Framework and Automated Clinical Insights using LLMs**

**📌 Project Overview**

This project focuses on the automated classification of skin diseases using a hybrid deep learning framework, combined with Large Language Models (LLMs) to provide clinical insights such as symptoms, causes, and treatments.

It aims to make early diagnosis more accurate, fast, and informative for users.

**🚀 Key Features**

Hybrid CNN model combining VGG16, MobileNetV3Small, and ResNet50 for better feature extraction.

Fine-tuning with frozen base layers to prevent overfitting.

Data Augmentation techniques to enhance dataset diversity.

LLM Integration to generate medical insights using models like ChatGPT, Copilot, Gemini, and Llama.

Cosine Similarity and Perplexity Scoring to evaluate and select the most accurate LLM responses.

Automatic retrieval of symptoms, causes, and treatments after disease prediction.

**🛠️ Tech Stack**

Python

TensorFlow / Keras

Scikit-Learn

Pandas

Matplotlib

Natural Language Processing (TF-IDF, Cosine Similarity)

**📂 Project Structure**

Folder/File	Description
model_training/	Scripts for building and training the hybrid CNN model

data/	Dataset files (train, validation, test)

llm_evaluation/	Code for evaluating LLM responses against Mayo references

clinical_info/	CSV file storing Gemini’s verified responses

README.md	Project overview and instructions

**🔥 How It Works**

Image Input: Upload a skin disease image.

Hybrid CNN Model: Classifies the disease into one of 8 classes (Bacterial, Fungal, Viral, Parasitic infections).

LLM Response: Fetches trusted medical insights about the predicted disease.

Display Results: Shows disease name, symptoms, causes, and treatments to the user.

**📊 Results**

High classification accuracy using feature fusion.

Gemini model was found most accurate (highest cosine similarity with Mayo Clinic references).

Complete system that performs both detection and explanation.
