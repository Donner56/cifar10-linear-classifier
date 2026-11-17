import os
import tarfile
import pickle

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 실험 설정
# ============================================================

DATA_PATH = "cifar-10-python.tar.gz"

RANDOM_SEED = 42

NUM_CLASSES = 10
IMAGE_SIZE = 3072

NUM_EXAMPLES = 6

# 가중치의 초기값을 작게 설정
WEIGHT_SCALE = 0.001

# SVM hinge loss의 margin
DELTA = 1.0

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# ============================================================
# 2. CIFAR-10 데이터 로드
# ============================================================

def load_cifar10(data_path):

    images = []
    labels = []

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"데이터셋 파일을 찾을 수 없습니다: {data_path}"
        )

    with tarfile.open(data_path, "r:gz") as tar:

        for i in range(1, 6):

            filename = (
                f"cifar-10-batches-py/data_batch_{i}"
            )

            file_obj = tar.extractfile(filename)

            batch = pickle.load(
                file_obj,
                encoding="bytes"
            )

            images.append(batch[b"data"])
            labels.extend(batch[b"labels"])

    X = np.concatenate(images, axis=0)
    y = np.array(labels)

    # float32 변환 및 정규화
    X = X.astype(np.float32) / 255.0

    print("데이터 로드 완료")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    return X, y


# ============================================================
# 3. 가중치 W와 편향 b 초기화
# ============================================================

def initialize_parameters():

    np.random.seed(RANDOM_SEED)

    # W: (10, 3072)
    W = (
        WEIGHT_SCALE
        * np.random.randn(NUM_CLASSES, IMAGE_SIZE)
    ).astype(np.float32)

    # b: (10,)
    b = np.zeros(NUM_CLASSES, dtype=np.float32)

    print("\n가중치 및 편향 초기화")
    print("W shape:", W.shape)
    print("b shape:", b.shape)

    print("W mean:", np.mean(W))
    print("W std:", np.std(W))
    print("b:", b)

    return W, b


# ============================================================
# 4. 선형 분류기 구현
# ============================================================

def linear_classifier(X, W, b):

    """
    f(x) = Wx + b

    X: (N, 3072)
    W: (10, 3072)
    b: (10,)

    반환:
    scores: (N, 10)
    """

    scores = np.dot(X, W.T) + b

    return scores


# ============================================================
# 5. Multiclass SVM Hinge Loss 구현
# ============================================================

def svm_hinge_loss(scores, y, delta=1.0):

    """
    Multiclass SVM Hinge Loss

    L_i = sum(j != y_i) max(0, s_j - s_yi + delta)

    전체 샘플의 평균 Loss 반환
    """

    num_samples = scores.shape[0]

    # 정답 클래스의 점수
    correct_scores = scores[
        np.arange(num_samples), y
    ]

    # 모든 클래스의 점수에서 정답 점수를 뺌
    margins = (
        scores
        - correct_scores[:, np.newaxis]
        + delta
    )

    # 정답 클래스의 margin은 0으로 설정
    margins[np.arange(num_samples), y] = 0

    # 음수 margin은 0으로 처리
    margins = np.maximum(0, margins)

    # 이미지별 Loss
    loss_per_image = np.sum(margins, axis=1)

    # 전체 이미지의 평균 Loss
    loss = np.mean(loss_per_image)

    return loss, loss_per_image


# ============================================================
# 6. Softmax 구현
# ============================================================

def softmax(scores):

    """
    Score를 확률로 변환한다.

    각 이미지의 클래스별 확률 합은 1이다.
    """

    # 수치적 안정성을 위해 최댓값을 빼줌
    shifted_scores = (
        scores - np.max(
            scores,
            axis=1,
            keepdims=True
        )
    )

    exp_scores = np.exp(shifted_scores)

    probabilities = (
        exp_scores
        / np.sum(
            exp_scores,
            axis=1,
            keepdims=True
        )
    )

    return probabilities


# ============================================================
# 7. 예시 이미지 선택
# ============================================================

def select_examples(X, y, W, b):

    """
    무작위 초기화 모델의 예측 결과를 확인하고
    정답 사례와 오답 사례를 선택한다.

    학습은 수행하지 않는다.
    """

    rng = np.random.default_rng(RANDOM_SEED)

    # 후보 이미지 2,000개 무작위 선택
    candidate_indices = rng.choice(
        len(X),
        size=2000,
        replace=False
    )

    X_candidates = X[candidate_indices]
    y_candidates = y[candidate_indices]

    scores = linear_classifier(
        X_candidates, W, b
    )

    predictions = np.argmax(scores, axis=1)

    correct_positions = np.where(
        predictions == y_candidates
    )[0]

    wrong_positions = np.where(
        predictions != y_candidates
    )[0]

    num_correct = min(
        NUM_EXAMPLES // 2,
        len(correct_positions)
    )

    num_wrong = min(
        NUM_EXAMPLES - num_correct,
        len(wrong_positions)
    )

    selected_positions = np.concatenate([
        correct_positions[:num_correct],
        wrong_positions[:num_wrong]
    ])

    selected_indices = candidate_indices[
        selected_positions
    ]

    return selected_indices


# ============================================================
# 8. 예시 이미지의 점수 및 확률 출력
# ============================================================

def show_predictions(X, y, W, b, selected_indices):

    X_selected = X[selected_indices]
    y_selected = y[selected_indices]

    # 선형 분류기 점수
    scores = linear_classifier(
        X_selected, W, b
    )

    # Softmax 확률
    probabilities = softmax(scores)

    # 가장 높은 점수의 클래스
    predictions = np.argmax(scores, axis=1)

    print("\n========== 예측 결과 ==========")

    for i in range(len(selected_indices)):

        print(f"\n[Image {i + 1}]")

        print(
            "실제 클래스:",
            CLASS_NAMES[y_selected[i]]
        )

        print(
            "예측 클래스:",
            CLASS_NAMES[predictions[i]]
        )

        print(
            "분류 결과:",
            "정답" if predictions[i] == y_selected[i]
            else "오답"
        )

        print("\n클래스별 Score 및 Probability")

        for c in range(NUM_CLASSES):

            print(
                f"{CLASS_NAMES[c]:12s} "
                f"Score={scores[i, c]:.6f} "
                f"Probability={probabilities[i, c]:.6f}"
            )

        print(
            "\n확률 합:",
            np.sum(probabilities[i])
        )

    return scores, probabilities, predictions


# ============================================================
# 9. SVM Loss 출력
# ============================================================

def show_svm_loss(scores, y, selected_indices):

    selected_labels = y[selected_indices]

    total_loss, individual_losses = svm_hinge_loss(
        scores,
        selected_labels,
        delta=DELTA
    )

    print("\n========== SVM Hinge Loss ==========")

    for i, loss in enumerate(individual_losses):

        print(
            f"Image {i + 1}: Loss = {loss:.6f}"
        )

    print(
        f"\n평균 SVM Loss: {total_loss:.6f}"
    )

    return total_loss


# ============================================================
# 10. 가중치 시각화
# ============================================================

def visualize_weights(W):

    """
    W의 각 클래스별 가중치 벡터를
    32 x 32 x 3 이미지로 변환한다.
    """

    fig, axes = plt.subplots(
        2, 5,
        figsize=(15, 7)
    )

    # 모든 클래스에 동일한 색상 범위 적용
    weight_images = W.reshape(
        NUM_CLASSES, 3, 32, 32
    ).transpose(0, 2, 3, 1)

    scale = np.max(np.abs(weight_images))

    for i, ax in enumerate(axes.flat):

        img = (
            weight_images[i] / (2 * scale) + 0.5
        )

        img = np.clip(img, 0, 1)

        ax.imshow(img)

        ax.set_title(CLASS_NAMES[i])
        ax.axis("off")

    plt.suptitle(
        "Class Weight Visualization (Random Initialization)",
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        "linear_classifier_weights.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 11. 예시 이미지 시각화
# ============================================================

def visualize_examples(
    X, y, selected_indices, predictions
):

    fig, axes = plt.subplots(
        2, 3,
        figsize=(12, 7)
    )

    for i, ax in enumerate(axes.flat):

        image = X[selected_indices[i]].reshape(
            3, 32, 32
        ).transpose(1, 2, 0)

        true_label = y[selected_indices[i]]
        predicted_label = predictions[i]

        ax.imshow(image)

        result = (
            "Correct"
            if true_label == predicted_label
            else "Wrong"
        )

        ax.set_title(
            f"True: {CLASS_NAMES[true_label]}\n"
            f"Pred: {CLASS_NAMES[predicted_label]} "
            f"({result})"
        )

        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        "linear_classifier_examples.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 12. Main
# ============================================================

if __name__ == "__main__":

    # 1. 데이터 로드
    X, y = load_cifar10(DATA_PATH)

    # 2. 가중치 및 편향 초기화
    W, b = initialize_parameters()

    # 3. 예시 이미지 선택
    selected_indices = select_examples(
        X, y, W, b
    )

    # 4. 예측 및 Softmax
    scores, probabilities, predictions = show_predictions(
        X, y, W, b, selected_indices
    )

    # 5. SVM Loss 계산
    svm_loss = show_svm_loss(
        scores, y, selected_indices
    )

    # 6. 가중치 시각화
    visualize_weights(W)

    # 7. 이미지 예측 결과 시각화
    visualize_examples(
        X, y, selected_indices, predictions
    )

    print("\n========== 실험 완료 ==========")
    print("모델 학습 및 가중치 업데이트는 수행하지 않았습니다.")
