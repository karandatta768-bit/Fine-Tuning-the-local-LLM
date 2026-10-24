# Dataset

The project downloads the Bitext Customer Support LLM Chatbot Training Dataset from the Hugging Face dataset repository:

- Repository: https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
- Dataset card/license: review the current dataset card and CDLA-Sharing-1.0 terms before redistribution or commercial use.
- Source CSV: `Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv`
- Expected fields used by preprocessing: `instruction`, `category`, `intent`, and `response`. The source also contains a `flags` field, which is not used for training.

The downloaded dataset is synthetic/curated, not verified live customer records. Generated raw and processed data are ignored by Git; obtain them by running the project scripts.
