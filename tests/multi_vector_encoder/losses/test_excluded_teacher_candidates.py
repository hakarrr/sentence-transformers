"""Explicit teacher exclusions must contribute zero KL, not 0 * -inf."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from sentence_transformers.multi_vector_encoder.losses import MultiVectorDistillKLDivLoss


class _IdentityEncoder(nn.Module):
    def forward(self, features, **kwargs):
        return features


@pytest.mark.parametrize("excluded", [True, False])
def test_teacher_exclusion_matches_probability_target_kl(excluded):
    teacher_temperature = 2.0
    generator = torch.Generator().manual_seed(31)
    embeddings = [torch.randn(2, 3, 4, generator=generator, requires_grad=True) for _ in range(4)]
    features = [{"token_embeddings": value, "attention_mask": torch.ones(2, 3)} for value in embeddings]
    labels = torch.tensor([[2.0, -float("inf") if excluded else -1.0, 0.5], [0.0, 1.0, 2.0]], requires_grad=True)
    loss = MultiVectorDistillKLDivLoss(
        _IdentityEncoder(), teacher_temperature=teacher_temperature, student_temperature=0.5
    )
    value = loss(features, labels)
    from sentence_transformers.multi_vector_encoder.scoring import colbert_kd_scores

    scores = colbert_kd_scores(embeddings[0], torch.stack(embeddings[1:], dim=1))
    expected = (
        torch.nn.functional.kl_div(
            (scores / 0.5).log_softmax(-1), (labels.detach() / teacher_temperature).softmax(-1), reduction="batchmean"
        )
        * 0.5**2
    )
    assert torch.isfinite(value)
    torch.testing.assert_close(value, expected)
    gradients = torch.autograd.grad(value, embeddings, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, embeddings)
    for actual, reference in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual, reference)
    assert labels.grad is None


@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("excluded", [True, False])
def test_evaluator_teacher_exclusion(nested, excluded):
    from unittest.mock import Mock

    from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorDistillationEvaluator

    class ScoreModel:
        model_card_data = Mock()

        def encode_query(self, inputs, **kwargs):
            return torch.ones(len(inputs), 1)

        def encode_document(self, inputs, **kwargs):
            return torch.tensor([[float(value)] for value in inputs])

        def similarity_pairwise(self, queries, documents):
            return documents[:, 0]

    teacher = [2.0, -float("inf") if excluded else -1.0, 0.5]
    evaluator = MultiVectorDistillationEvaluator(
        queries=["q"] if nested else ["q"] * 3,
        documents=[["2", "1", "0.5"]] if nested else ["2", "1", "0.5"],
        scores=[teacher] if nested else teacher,
        write_csv=False,
    )
    actual = evaluator(ScoreModel())["kl_divergence"]
    expected = torch.nn.functional.kl_div(
        torch.tensor([2.0, 1.0, 0.5]).log_softmax(-1), torch.tensor(teacher).softmax(-1), reduction="sum"
    )
    assert actual == pytest.approx(expected.item())


def test_finite_teacher_target_is_bit_exact_with_original_log_target_kl():
    from sentence_transformers.multi_vector_encoder.scoring import colbert_kd_scores

    generator = torch.Generator().manual_seed(41)
    embeddings = [torch.randn(2, 3, 4, generator=generator) for _ in range(3)]
    features = [{"token_embeddings": value, "attention_mask": torch.ones(2, 3)} for value in embeddings]
    labels = torch.tensor([[2.0, -1.0], [1.0, -1000.0]])
    scores = colbert_kd_scores(embeddings[0], torch.stack(embeddings[1:], dim=1))
    expected = torch.nn.functional.kl_div(
        scores.log_softmax(-1), labels.log_softmax(-1), reduction="batchmean", log_target=True
    )
    actual = MultiVectorDistillKLDivLoss(_IdentityEncoder())(features, labels)
    assert torch.equal(actual, expected)


def test_nan_teacher_remains_an_error():
    features = [{"token_embeddings": torch.ones(1, 2, 3), "attention_mask": torch.ones(1, 2)} for _ in range(3)]
    with pytest.raises(ValueError, match="NaN"):
        MultiVectorDistillKLDivLoss(_IdentityEncoder())(features, torch.tensor([[0.0, float("nan")]]))


def test_excluded_teacher_training_and_evaluation_agree(mve_bert_tiny_model):
    from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorDistillationEvaluator

    model = mve_bert_tiny_model
    model.eval()
    queries = ["capital of France", "capital of Germany"]
    documents = [
        ["Paris is the capital of France", "Berlin is the capital of Germany"],
        ["Berlin is in Germany", "Paris is in France"],
    ]
    labels = torch.tensor([[2.0, -float("inf")], [2.0, 0.0]], device=model.device)
    features = [model.preprocess(queries, task="query")]
    features.extend(model.preprocess([row[index] for row in documents], task="document") for index in range(2))
    loss = MultiVectorDistillKLDivLoss(model, student_temperature=0.5, teacher_temperature=2.0)
    value = loss(features, labels)
    assert torch.isfinite(value)
    value.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    evaluator = MultiVectorDistillationEvaluator(
        queries, documents, labels.cpu(), student_temperature=0.5, teacher_temperature=2.0, write_csv=False
    )
    metric = evaluator(model)["kl_divergence"]
    assert metric == pytest.approx(value.item(), rel=1e-4, abs=1e-6)
