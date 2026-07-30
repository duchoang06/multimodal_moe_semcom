#!/usr/bin/env python3
import os
import re
import json
import argparse
from collections import Counter
from typing import Dict, List, Tuple


def normalize_answer(ans: str) -> str:
    """
    Light VQA-style normalization.
    Keeps things simple for classification:
      - lowercase
      - strip spaces
      - remove punctuation except digits/letters/spaces
      - collapse whitespace
    """
    ans = ans.lower().strip()
    ans = re.sub(r"[^a-z0-9\s]", "", ans)
    ans = re.sub(r"\s+", " ", ans).strip()
    return ans


def majority_answer(annotation: dict) -> str:
    """
    VQAv2 has 10 human answers per question.
    We use the most frequent normalized answer as the single-label target.
    """
    answers = [normalize_answer(a["answer"]) for a in annotation["answers"]]
    counts = Counter(answers)
    return counts.most_common(1)[0][0]


def load_questions(path: str) -> List[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    return data["questions"]


def load_annotations(path: str) -> List[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    return data["annotations"]


def build_answer_vocab(train_annotations: List[dict], top_k: int) -> Tuple[Dict[str, int], Dict[int, str], Counter]:
    """
    Build answer vocab from TRAIN split only.
    """
    counter = Counter()

    for ann in train_annotations:
        answers = [normalize_answer(a["answer"]) for a in ann["answers"]]
        counter.update(answers)

    most_common = counter.most_common(top_k)
    ans2id = {ans: i for i, (ans, _) in enumerate(most_common)}
    id2ans = {i: ans for ans, i in ans2id.items()}
    return ans2id, id2ans, counter

def answer_scores(answers: List[str], ans2id: Dict[str, int]) -> List[float]:
    """
    Convert the 10 VQAv2 answers into the soft target vector used by BCE.
    """
    scores = [0.0] * len(ans2id)
    counts = Counter(normalize_answer(ans) for ans in answers)

    for ans, occur in counts.items():
        ans_id = ans2id.get(ans)
        if ans_id is None:
            continue

        if occur == 0:
            score = 0.0
        elif occur == 1:
            score = 0.3
        elif occur == 2:
            score = 0.6
        elif occur == 3:
            score = 0.9
        else:
            score = 1.0

        scores[ans_id] = score

    return scores

# def build_samples_old(
#     questions: List[dict],
#     annotations: List[dict],
#     ans2id: Dict[str, int],
# ) -> List[dict]:
#     """
#     Build filtered sample list:
#       {
#         question_id,
#         image_id,
#         question,
#         answer,
#         answer_id
#       }
#     Keeps only samples whose majority answer is in ans2id.
#     """
#     qid2q = {q["question_id"]: q for q in questions}

#     samples = []
#     skipped_missing_question = 0
#     skipped_oov_answer = 0

#     for ann in annotations:
#         qid = ann["question_id"]
#         q = qid2q.get(qid)

#         if q is None:
#             skipped_missing_question += 1
#             continue

#         ans = majority_answer(ann)
#         if ans not in ans2id:
#             skipped_oov_answer += 1
#             continue

#         samples.append(
#             {
#                 "question_id": qid,
#                 "image_id": q["image_id"],
#                 "question": q["question"],
#                 "answer": ans,
#                 "answer_id": ans2id[ans],
#             }
#         )

#     print(f"  built samples: {len(samples)}")
#     print(f"  skipped (missing question): {skipped_missing_question}")
#     print(f"  skipped (answer not in vocab): {skipped_oov_answer}")
#     return samples

def build_samples(
    questions: List[dict],
    annotations: List[dict],
    ans2id: Dict[str, int],
) -> List[dict]:
    
    qid2q = {q["question_id"]: q for q in questions}

    samples = []
    skipped_missing_question = 0
    skipped_oov_answer = 0

    for ann in annotations:
        qid = ann["question_id"]
        q = qid2q.get(qid)

        if q is None:
            skipped_missing_question += 1
            continue

        normalized_answers = [normalize_answer(a["answer"]) for a in ann["answers"]]

        ans = majority_answer(ann)
        if ans not in ans2id:
            skipped_oov_answer += 1
            continue

        samples.append(
            {
                "question_id": qid,
                "image_id": q["image_id"],
                "question": q["question"],
                "answer": ans,
                "answers": normalized_answers,
                "answer_id": ans2id[ans],

                # "answer_scores": answer_scores(normalized_answers, ans2id),
            }
        )

    print(f"  built samples: {len(samples)}")
    print(f"  skipped (missing question): {skipped_missing_question}")
    print(f"  skipped (answer not in vocab): {skipped_oov_answer}")
    return samples

def verify_expected_files(root: str) -> None:
    required = [
        "v2_OpenEnded_mscoco_train2014_questions.json",
        "v2_OpenEnded_mscoco_val2014_questions.json",
        "v2_mscoco_train2014_annotations.json",
        "v2_mscoco_val2014_annotations.json",
        "train2014",
        "val2014",
    ]

    missing = [name for name in required if not os.path.exists(os.path.join(root, name))]
    if missing:
        raise FileNotFoundError(
            "Missing required VQAv2 files/folders under root:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )


def save_json(obj, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f)


def main():
    parser = argparse.ArgumentParser(description="Prepare cached VQAv2 classification metadata.")
    parser.add_argument("--root", type=str, required=True, help="Path to VQAv2 root directory.")
    parser.add_argument("--top_k", type=int, default=3000, help="Top-K answers to keep.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output cache directory. Defaults to <root>/cache",
    )
    args = parser.parse_args()

    root = args.root
    top_k = args.top_k
    out_dir = args.out_dir or os.path.join(root, "cache")
    os.makedirs(out_dir, exist_ok=True)

    verify_expected_files(root)

    train_q_path = os.path.join(root, "v2_OpenEnded_mscoco_train2014_questions.json")
    val_q_path = os.path.join(root, "v2_OpenEnded_mscoco_val2014_questions.json")
    train_a_path = os.path.join(root, "v2_mscoco_train2014_annotations.json")
    val_a_path = os.path.join(root, "v2_mscoco_val2014_annotations.json")

    print("Loading raw VQAv2 files...")
    train_questions = load_questions(train_q_path)
    val_questions = load_questions(val_q_path)
    train_annotations = load_annotations(train_a_path)
    val_annotations = load_annotations(val_a_path)

    print(f"Train questions: {len(train_questions)}")
    print(f"Val questions:   {len(val_questions)}")
    print(f"Train annots:    {len(train_annotations)}")
    print(f"Val annots:      {len(val_annotations)}")

    print(f"\nBuilding top-{top_k} answer vocab from TRAIN annotations...")
    ans2id, id2ans, answer_counter = build_answer_vocab(train_annotations, top_k=top_k)

    print(f"Answer vocab size: {len(ans2id)}")
    print("Top 20 answers:")
    for ans, freq in answer_counter.most_common(20):
        print(f"  {ans!r}: {freq}")

    print("\nBuilding TRAIN samples...")
    train_samples = build_samples(train_questions, train_annotations, ans2id)

    print("\nBuilding VAL samples...")
    val_samples = build_samples(val_questions, val_annotations, ans2id)

    ans2id_path = os.path.join(out_dir, f"ans2id_top{top_k}.json")
    id2ans_path = os.path.join(out_dir, f"id2ans_top{top_k}.json")
    train_samples_path = os.path.join(out_dir, f"train_samples_top{top_k}.json")
    val_samples_path = os.path.join(out_dir, f"val_samples_top{top_k}.json")

    print("\nSaving cache files...")
    save_json(ans2id, ans2id_path)
    save_json(id2ans, id2ans_path)
    save_json(train_samples, train_samples_path)
    save_json(val_samples, val_samples_path)

    print("\nDone.")
    print(f"Saved: {ans2id_path}")
    print(f"Saved: {id2ans_path}")
    print(f"Saved: {train_samples_path}")
    print(f"Saved: {val_samples_path}")


if __name__ == "__main__":
    main()
