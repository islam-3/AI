# Fine-Tuning GPT-2 on Custom Text Data

## 📌 Project Overview

This project fine-tunes the GPT-2 small language model on a small custom dataset of personal reflections and motivational thoughts. After training, the model can generate new text that mimics the style and tone of the original data.

## 📁 Files Included

- `custom_data.txt`: The personal dataset used for fine-tuning.
- `train_gpt2.py`: Python script for fine-tuning GPT-2.
- `README.md`: This documentation file.

## ⚙️ How to Run

### Step 1: Install Required Libraries

```bash
pip install transformers datasets
```

### Step 2: Fine-Tune the Model

```bash
python train_gpt2.py
```

This will fine-tune GPT-2 on your `custom_data.txt` and save the new model to the `gpt2-finetuned` directory.

## 🧠 Model Used

- Pretrained Model: `gpt2` (from Hugging Face 🤗 Transformers)
- Fine-tuning on: 20 short personal writings

## 📝 Notes

- This is a lightweight demo project, ideal for students and educational purposes.
- Training was tested on a small dataset. For better performance, you can add more data.
- GPU is recommended for faster training.

## 👨‍💻 Author

Project developed for academic purposes by Eslem Hasan.  
Student ID: 220208952
