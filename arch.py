import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# =========================================================
# FIGURE SETUP
# =========================================================

fig, ax = plt.subplots(figsize=(10, 18))

ax.set_xlim(0, 10)
ax.set_ylim(0, 20)

ax.axis("off")


# =========================================================
# BOX FUNCTION
# =========================================================

def draw_box(
    x, y, width, height,
    title,
    content="",
    edgecolor="#4F9FC1",
    facecolor="white",
    title_size=13,
    content_size=10,
    title_weight="bold"
):

    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.22",
        linewidth=2.0,
        edgecolor=edgecolor,
        facecolor=facecolor
    )

    ax.add_patch(box)

    # Title
    ax.text(
        x + width / 2,
        y + height * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight=title_weight
    )

    # Content
    if content:

        ax.text(
            x + width / 2,
            y + height * 0.34,
            content,
            ha="center",
            va="center",
            fontsize=content_size,
            linespacing=1.5
        )


# =========================================================
# ARROW FUNCTION
# =========================================================

def draw_arrow(x1, y1, x2, y2):

    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.8,
        color="black"
    )

    ax.add_patch(arrow)


# =========================================================
# 1. CT IMAGE DATA
# =========================================================

draw_box(
    1.0, 18.4, 8.0, 0.9,
    "CT Image Dataset",
    "Normal  |  Cyst  |  Tumor  |  Stone",
    edgecolor="#4F81BD",
    facecolor="#EAF2F8",
    title_size=14,
    content_size=10
)


# =========================================================
# 2. IMAGE PREPROCESSING
# =========================================================

draw_box(
    1.0, 16.5, 8.0, 1.1,
    "Image Preprocessing",
    "Resize\nGrayscale Conversion",
    edgecolor="#45A6B8",
    facecolor="#E8F7F8",
    title_size=13,
    content_size=10
)


# =========================================================
# 3. MULTI-FILTER ENHANCEMENT
# =========================================================

draw_box(
    0.8, 14.5, 8.4, 1.2,
    "Multi-Filter Image Enhancement",
    "CLAHE  |  Gaussian  |  Median  |  Sharpened  |  Bilateral",
    edgecolor="#82B44C",
    facecolor="#EFF8E6",
    title_size=13,
    content_size=10
)


# =========================================================
# 4. ENHANCED CT IMAGES
# =========================================================

draw_box(
    1.0, 12.8, 8.0, 0.9,
    "Enhanced CT Images",
    "Five Filtered Image Sets",
    edgecolor="#4F81BD",
    facecolor="#EAF2F8",
    title_size=13,
    content_size=10
)


# =========================================================
# 5. GLCM FEATURE EXTRACTION
# =========================================================

glcm_text = (
    "Contrast  |  Dissimilarity  |  Homogeneity  |  ASM\n"
    "Energy  |  Correlation  |  Entropy  |  Mean\n"
    "Variance  |  Cluster Shade  |  Cluster Prominence\n"
    "Maximum Probability"
)

draw_box(
    0.6, 10.3, 8.8, 1.7,
    "GLCM Texture Feature Extraction and Analysis",
    glcm_text,
    edgecolor="#C49A28",
    facecolor="#FFF7DD",
    title_size=12,
    content_size=9
)


# =========================================================
# 6. FEATURE REPRESENTATION
# =========================================================

draw_box(
    1.0, 8.7, 8.0, 0.9,
    "Feature Representation",
    "14-Dimensional GLCM Feature Vector",
    edgecolor="#9A70C2",
    facecolor="#F3EAFB",
    title_size=13,
    content_size=10
)


# =========================================================
# 7. ML CLASSIFIERS
# =========================================================

draw_box(
    1.0, 7.0, 8.0, 1.0,
    "Machine Learning Classifiers",
    "Random Forest  |  XGBoost  |  KNN  |  SVM  |  Logistic Regression",
    edgecolor="#D4779F",
    facecolor="#FCEAF2",
    title_size=13,
    content_size=9.5
)


# =========================================================
# 8. PERFORMANCE EVALUATION
# =========================================================

draw_box(
    1.0, 5.3, 8.0, 1.0,
    "Performance Evaluation",
    "Accuracy  |  Precision  |  Recall  |  F1-Score",
    edgecolor="#4F81BD",
    facecolor="#EAF2F8",
    title_size=13,
    content_size=10
)


# =========================================================
# 9. OPTIMAL PAIR
# =========================================================

draw_box(
    0.8, 3.4, 8.4, 1.2,
    "Optimal Filter–Classifier Pair",
    "Sharpened Image Enhancement + Random Forest Classifier",
    edgecolor="#6E9F35",
    facecolor="#E9F5D8",
    title_size=13,
    content_size=11
)


# =========================================================
# 10. FINAL CLASSIFICATION
# =========================================================

draw_box(
    1.0, 1.5, 8.0, 1.0,
    "Kidney CT Classification",
    "Normal  |  Cyst  |  Tumor  |  Stone",
    edgecolor="#D88A2D",
    facecolor="#FFF0DD",
    title_size=14,
    content_size=10
)


# =========================================================
# ARROWS
# =========================================================

draw_arrow(5, 18.4, 5, 17.6)

draw_arrow(5, 16.5, 5, 15.7)

draw_arrow(5, 14.5, 5, 13.7)

draw_arrow(5, 12.8, 5, 12.0)

draw_arrow(5, 10.3, 5, 9.6)

draw_arrow(5, 8.7, 5, 8.0)

draw_arrow(5, 7.0, 5, 6.3)

draw_arrow(5, 5.3, 5, 4.6)

draw_arrow(5, 3.4, 5, 2.5)


# =========================================================
# SAVE HIGH-QUALITY FIGURES
# =========================================================

plt.savefig(
    "Kidney_CT_Classification_Architecture.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

plt.savefig(
    "Kidney_CT_Classification_Architecture.pdf",
    dpi=600,
    bbox_inches="tight"
)


# =========================================================
# DISPLAY
# =========================================================

plt.show()
