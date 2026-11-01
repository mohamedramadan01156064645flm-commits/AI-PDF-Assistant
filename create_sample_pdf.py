"""Utility script to generate a sample multi-page bilingual PDF document
(machine_learning_basics.pdf) for immediate testing and demonstration.
"""

from pathlib import Path
try:
    import pymupdf  # PyMuPDF 2.0+
except ImportError:
    import fitz as pymupdf
from config import PDFS_DIR


def generate_sample_pdf(output_path: Path = PDFS_DIR / "machine_learning_basics.pdf") -> Path:
    """Creates a sample 3-page Machine Learning textbook in PDF format."""
    doc = pymupdf.open()

    # Use Arial if available for proper Unicode/Arabic rendering
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font_kwargs = {"fontfile": str(font_path)} if font_path.exists() else {"fontname": "helv"}

    # --- Page 1: Introduction to Machine Learning ---
    page1 = doc.new_page()
    text_page1 = """MACHINE LEARNING ESSENTIALS & FOUNDATIONS
Chapter 1: Overview of Machine Learning Paradigms

1.1 Introduction
Machine Learning (ML) is a subset of Artificial Intelligence (AI) that empowers computer systems to learn from data and improve their performance on a specific task without being explicitly programmed.

1.2 Supervised Learning vs. Unsupervised Learning
- Supervised Learning:
In supervised learning, the algorithm is trained on a labeled dataset, meaning each training example is paired with an input and an expected target label or ground-truth outcome.
Key tasks in supervised learning include classification (e.g., predicting spam vs. ham email, disease diagnosis) and regression (e.g., predicting house prices, stock market trends). Popular algorithms include Logistic Regression, Support Vector Machines (SVM), Random Forests, and Gradient Boosted Trees.

- Unsupervised Learning:
In unsupervised learning, the model works with unlabeled data. The objective is to discover inherent patterns, hidden groupings, or latent representations within the dataset without human intervention or predefined categories.
Key tasks include clustering (e.g., K-Means, DBSCAN for customer segmentation) and dimensionality reduction (e.g., Principal Component Analysis / PCA, t-SNE).

1.3 Summary & Arabic Overview:
التعلم الآلي (Machine Learning) هو فرع من الذكاء الاصطناعي.
الفرق بين التعلم الخاضع للإشراف (Supervised Learning) والتعلم غير الخاضع للإشراف (Unsupervised Learning) يكمن في وجود البيانات المصنفة بتسميات محددة.
"""
    page1.insert_text((50, 50), text_page1, fontsize=10, **font_kwargs)

    # --- Page 2: Model Evaluation & Overfitting ---
    page2 = doc.new_page()
    text_page2 = """MACHINE LEARNING ESSENTIALS & FOUNDATIONS
Chapter 2: Model Evaluation and Generalization

2.1 Classification Evaluation Metrics
Evaluating a machine learning model requires metrics that reflect both positive and negative prediction quality:
1. Accuracy: Ratio of correct predictions over total predictions. Often misleading when classes are imbalanced.
2. Precision: Out of all instances predicted as positive, how many were truly positive? (TP / (TP + FP)).
3. Recall (Sensitivity): Out of all actual positive instances, how many did the model identify? (TP / (TP + FN)).
4. F1-Score: The harmonic mean of precision and recall (2 * (Precision * Recall) / (Precision + Recall)).
5. ROC-AUC: Receiver Operating Characteristic Area Under Curve, measuring discriminative power across decision thresholds.

2.2 Overfitting and Prevention Techniques
Overfitting occurs when a statistical model or machine learning algorithm captures noise, fluctuations, and idiosyncratic patterns in the training data rather than the underlying general trend. As a result, the model displays near-perfect accuracy on training data but fails to generalize to unseen test data.

Methods to prevent and mitigate overfitting:
- Cross-Validation: Utilizing k-fold cross-validation to assess variance across data partitions.
- Regularization: Adding penalization terms such as L1 (Lasso) and L2 (Ridge) to constrain model weights.
- Dropout: Randomly deactivating a subset of neurons during each training step in deep neural networks.
- Early Stopping: Halting training when validation loss stops improving.
"""
    page2.insert_text((50, 50), text_page2, fontsize=10, **font_kwargs)

    # --- Page 3: Deep Learning and Neural Networks ---
    page3 = doc.new_page()
    text_page3 = """MACHINE LEARNING ESSENTIALS & FOUNDATIONS
Chapter 3: Deep Learning Architectures and Modern NLP

3.1 What is Deep Learning?
Deep Learning is a specialized subfield of machine learning based on artificial neural networks with multiple representation layers (hence 'deep'). Unlike traditional machine learning models that depend heavily on manual feature engineering, deep neural networks automatically learn hierarchical feature representations directly from raw inputs.

3.2 Transformer Architecture and RAG Systems
Modern NLP and generative language models are built on the Transformer architecture introduced by Vaswani et al. in 2017. Transformers leverage multi-head self-attention mechanisms to process tokens in parallel, capturing long-range contextual relationships.

Retrieval-Augmented Generation (RAG) combines dense semantic retrieval with parametric LLM generation. By dynamically searching external vector databases (such as ChromaDB) using embedding models (such as sentence-transformers), RAG provides models with factual, up-to-date context, virtually eliminating factual hallucinations and enabling exact page citations.

شبكات التعلم العميق (Deep Learning) تعتمد على طبقات عصبية متعددة لاستخراج الميزات تلقائياً من البيانات الضخمة.
"""
    page3.insert_text((50, 50), text_page3, fontsize=10, **font_kwargs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    doc.close()
    print(f"✅ Generated sample PDF at: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_sample_pdf()
