"""
Smoke tests: 验证 ToMe patch 在 timm 0.9.16 下正常工作。
服务器运行：pytest tests/test_tome_patch.py -v
"""
import torch
import timm
import tome


def test_patch_applies_without_error():
    """ToMe patch 应能无报错地应用到 timm 模型。"""
    model = timm.create_model('deit_small_patch16_224', pretrained=False)
    tome.patch.timm(model, prop_attn=False)
    model.r = 8
    assert hasattr(model, '_tome_info'), "_tome_info 应在 patch 后存在"
    assert model._tome_info['class_token'] is True


def test_forward_r0_correct_shape():
    """r=0（不合并）时，输出 shape 应为 (batch, 1000)，无 NaN。"""
    model = timm.create_model('deit_small_patch16_224', pretrained=False)
    tome.patch.timm(model, prop_attn=False)
    model.r = 0
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 1000), f"期望 (2, 1000)，实际 {out.shape}"
    assert not torch.isnan(out).any(), "输出包含 NaN"
    assert not torch.isinf(out).any(), "输出包含 Inf"


def test_forward_r16_correct_shape():
    """r=16 时，输出 shape 应为 (batch, 1000)，无 NaN。"""
    model = timm.create_model('deit_small_patch16_224', pretrained=False)
    tome.patch.timm(model, prop_attn=False)
    model.r = 16
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 1000), f"期望 (2, 1000)，实际 {out.shape}"
    assert not torch.isnan(out).any(), "输出包含 NaN"
    assert not torch.isinf(out).any(), "输出包含 Inf"


def test_prop_attn_no_error():
    """prop_attn=True（离线评估模式）也应正常工作。"""
    model = timm.create_model('deit_small_patch16_224', pretrained=False)
    tome.patch.timm(model, prop_attn=True)
    model.r = 8
    model.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 1000)
    assert not torch.isnan(out).any()


def test_source_tracking():
    """trace_source=True 时，_tome_info['source'] 应被填充。"""
    model = timm.create_model('deit_tiny_patch16_224', pretrained=False)
    tome.patch.timm(model, trace_source=True, prop_attn=False)
    model.r = 8
    model.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        _ = model(x)

    source = model._tome_info.get('source')
    assert source is not None, "source tracking 应返回非 None"
    assert source.dim() == 3, f"source 应为 3D tensor，实际 {source.dim()}D"
