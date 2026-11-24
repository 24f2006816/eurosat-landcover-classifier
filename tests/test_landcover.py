import torch
from torch.utils.data import DataLoader, TensorDataset

from landcover.data import stratified_split
from landcover.models import build, count_parameters
from landcover.train import evaluate, fit, latency_ms


def test_stratified_split_keeps_class_balance_and_no_overlap():
    targets = [i % 10 for i in range(1000)]
    train, val, test = stratified_split(targets, seed=0)
    assert len(train) == 700 and len(val) == 150 and len(test) == 150
    assert not set(train) & set(val) and not set(train) & set(test) and not set(val) & set(test)
    for part in (train, val, test):
        counts = [sum(targets[i] == c for i in part) for c in range(10)]
        assert max(counts) == min(counts)


def test_split_is_deterministic():
    targets = [i % 3 for i in range(90)]
    assert stratified_split(targets, seed=1) == stratified_split(targets, seed=1)


def test_models_output_ten_logits():
    for name, size in [("cnn-scratch", 64), ("mobilenet-frozen", 128), ("mobilenet-finetune", 128)]:
        model = build(name, pretrained=False).eval()
        assert model(torch.randn(2, 3, size, size)).shape == (2, 10)


def test_frozen_backbone_trains_only_the_head():
    total, trainable = count_parameters(build("mobilenet-frozen", pretrained=False))
    assert trainable < total * 0.6
    _, all_trainable = count_parameters(build("mobilenet-finetune", pretrained=False))
    assert all_trainable == total


def test_training_loop_learns_a_separable_toy_problem():
    torch.manual_seed(0)
    x = torch.randn(200, 3, 32, 32)
    y = (x[:, 0].mean(dim=(1, 2)) > 0).long()  # label = sign of the red channel's mean
    x[:, 0] += (y.float() * 2 - 1)[:, None, None]  # make it easy
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)
    model = build("cnn-scratch", num_classes=2)
    info = fit(model, loader, loader, epochs=4, lr=5e-3, log=lambda *_: None)
    metrics = evaluate(model, loader, ["neg", "pos"])
    assert info["best_val_acc"] > 0.9 and metrics["accuracy"] > 0.9
    assert sum(map(sum, metrics["confusion_matrix"])) == 200


def test_latency_measurement_runs():
    assert latency_ms(build("cnn-scratch").eval(), 64, runs=2) > 0
