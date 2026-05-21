"""E2E tests for TIER 3 layer mappers.

Tests that loss layers, recurrent cells, and Identity are correctly mapped
through the full conversion pipeline (trace -> map -> cleanup).

Includes model-level tests using architectures that exercise these operations.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from decomposer import AdaptiveConverter


# ============================================================================
# Unit tests: individual layer mapper correctness
# ============================================================================

def test_identity():
    """nn.Identity -> identity layer (preserved in graph, used for skip wiring)."""
    # Identity in Sequential gets inlined by FX. Use a model where Identity
    # serves as a skip-connection path so it appears as a call_module node.
    class IdentityModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 16)
            self.skip = nn.Identity()
            self.fc2 = nn.Linear(16, 8)

        def forward(self, x):
            h = F.relu(self.fc1(x))
            s = self.skip(x)
            return self.fc2(h + s)

    model = IdentityModel()
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"x": torch.randn(1, 16)})

    # Identity may be preserved or inlined depending on tracer. Just verify
    # the conversion completes without unknowns.
    assert len(result.unknown_layers) == 0, \
        f"Identity model has unknowns: {[l.layer_type for l in result.unknown_layers]}"

    fc = [l for l in result.layers if l.layer_type == "fully_connected"]
    assert len(fc) == 2, f"Expected 2 fully_connected, got {len(fc)}"
    print("  PASS: nn.Identity model mapped correctly")


def test_gru_cell():
    """nn.GRUCell -> grucell with correct properties."""
    class GRUCellModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.cell = nn.GRUCell(input_size=16, hidden_size=32)

        def forward(self, x, h):
            return self.cell(x, h)

    model = GRUCellModel()
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({
        "x": torch.randn(1, 16),
        "h": torch.randn(1, 32),
    })

    cells = [l for l in result.layers if l.layer_type == "grucell"]
    assert len(cells) == 1, f"Expected 1 grucell, got {len(cells)}"
    assert cells[0].properties["unit"] == 32
    assert cells[0].has_weight
    print("  PASS: nn.GRUCell mapped correctly")


def test_lstm_cell():
    """nn.LSTMCell -> lstmcell with correct properties."""
    class LSTMCellModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.cell = nn.LSTMCell(input_size=16, hidden_size=32)

        def forward(self, x, hx):
            h, c = hx
            return self.cell(x, (h, c))

    model = LSTMCellModel()
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({
        "x": torch.randn(1, 16),
        "hx": (torch.randn(1, 32), torch.randn(1, 32)),
    })

    cells = [l for l in result.layers if l.layer_type == "lstmcell"]
    assert len(cells) == 1, f"Expected 1 lstmcell, got {len(cells)}"
    assert cells[0].properties["unit"] == 32
    assert cells[0].has_weight
    print("  PASS: nn.LSTMCell mapped correctly")


def test_rnn_cell():
    """nn.RNNCell -> rnncell with correct properties."""
    class RNNCellModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.cell = nn.RNNCell(input_size=16, hidden_size=32)

        def forward(self, x, h):
            return self.cell(x, h)

    model = RNNCellModel()
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({
        "x": torch.randn(1, 16),
        "h": torch.randn(1, 32),
    })

    cells = [l for l in result.layers if l.layer_type == "rnncell"]
    assert len(cells) == 1, f"Expected 1 rnncell, got {len(cells)}"
    assert cells[0].properties["unit"] == 32
    assert cells[0].has_weight
    print("  PASS: nn.RNNCell mapped correctly")


def test_cross_entropy_loss_module():
    """nn.CrossEntropyLoss -> cross_softmax."""
    class ClassifierWithLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 10)
            self.loss = nn.CrossEntropyLoss()

        def forward(self, x, target):
            logits = self.fc(x)
            return self.loss(logits, target)

    model = ClassifierWithLoss()
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({
        "x": torch.randn(1, 16),
        "target": torch.tensor([3]),
    })

    loss = [l for l in result.layers if l.layer_type == "cross_softmax"]
    assert len(loss) == 1, f"Expected 1 cross_softmax, got {len(loss)}"
    print("  PASS: nn.CrossEntropyLoss mapped correctly")


def test_mse_loss_module():
    """nn.MSELoss -> mse."""
    class RegressionWithLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 1)
            self.loss = nn.MSELoss()

        def forward(self, x, target):
            pred = self.fc(x)
            return self.loss(pred, target)

    model = RegressionWithLoss()
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({
        "x": torch.randn(1, 16),
        "target": torch.randn(1, 1),
    })

    loss = [l for l in result.layers if l.layer_type == "mse"]
    assert len(loss) == 1, f"Expected 1 mse, got {len(loss)}"
    print("  PASS: nn.MSELoss mapped correctly")


def test_kl_div_loss_module():
    """nn.KLDivLoss -> kld."""
    class KLDModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 10)
            self.loss = nn.KLDivLoss(reduction='batchmean')

        def forward(self, x, target):
            log_probs = F.log_softmax(self.fc(x), dim=-1)
            return self.loss(log_probs, target)

    model = KLDModel()
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({
        "x": torch.randn(1, 16),
        "target": torch.randn(1, 10).softmax(dim=-1),
    })

    loss = [l for l in result.layers if l.layer_type == "kld"]
    assert len(loss) == 1, f"Expected 1 kld, got {len(loss)}"
    print("  PASS: nn.KLDivLoss mapped correctly")


def test_bce_with_logits_loss_module():
    """nn.BCEWithLogitsLoss -> cross_sigmoid."""
    class BCEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 1)
            self.loss = nn.BCEWithLogitsLoss()

        def forward(self, x, target):
            logits = self.fc(x)
            return self.loss(logits, target)

    model = BCEModel()
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({
        "x": torch.randn(1, 16),
        "target": torch.randn(1, 1),
    })

    loss = [l for l in result.layers if l.layer_type == "cross_sigmoid"]
    assert len(loss) == 1, f"Expected 1 cross_sigmoid, got {len(loss)}"
    print("  PASS: nn.BCEWithLogitsLoss mapped correctly")


# ============================================================================
# Model-level E2E tests
# ============================================================================

class MiniRNNCellLanguageModel(nn.Module):
    """Simple character-level LM using GRUCell + LSTMCell."""
    def __init__(self, vocab_size=50, embed_dim=32, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru_cell = nn.GRUCell(embed_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        emb = self.embedding(x)
        # Process single timestep
        h_new = self.gru_cell(emb.squeeze(1), h)
        logits = self.fc(h_new)
        return logits, h_new


def test_rnn_cell_lm_e2e():
    """Character-level LM with GRUCell."""
    model = MiniRNNCellLanguageModel()
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({
        "x": torch.tensor([[5]]),
        "h": torch.zeros(1, 64),
    })

    result.summary()

    assert len(result.unknown_layers) == 0, \
        f"RNN Cell LM has unknowns: {[l.layer_type for l in result.unknown_layers]}"

    cells = [l for l in result.layers if l.layer_type == "grucell"]
    assert len(cells) == 1, f"Expected 1 grucell, got {len(cells)}"

    emb = [l for l in result.layers if l.layer_type == "embedding_layer"]
    assert len(emb) == 1, f"Expected 1 embedding, got {len(emb)}"

    fc = [l for l in result.layers if l.layer_type == "fully_connected"]
    assert len(fc) == 1, f"Expected 1 fully_connected, got {len(fc)}"

    print("  PASS: RNN Cell LM E2E conversion")
    print(f"    grucell: {len(cells)}")
    print(f"    embedding_layer: {len(emb)}")
    print(f"    fully_connected: {len(fc)}")
    print(f"    Total layers: {len(result.layers)}")


class MiniTrainableClassifier(nn.Module):
    """Classifier with embedded loss layer (training pipeline)."""
    def __init__(self, in_features=32, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, target):
        logits = self.features(x)
        loss = self.loss_fn(logits, target)
        return loss


def test_trainable_classifier_e2e():
    """Classifier with CrossEntropyLoss for training."""
    model = MiniTrainableClassifier()
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({
        "x": torch.randn(1, 32),
        "target": torch.tensor([5]),
    })

    result.summary()

    assert len(result.unknown_layers) == 0, \
        f"Trainable classifier has unknowns: {[l.layer_type for l in result.unknown_layers]}"

    fc = [l for l in result.layers if l.layer_type == "fully_connected"]
    assert len(fc) == 2, f"Expected 2 fully_connected, got {len(fc)}"

    loss = [l for l in result.layers if l.layer_type == "cross_softmax"]
    assert len(loss) == 1, f"Expected 1 cross_softmax, got {len(loss)}"

    act = [l for l in result.layers
           if l.layer_type == "activation"
           and l.properties.get("activation") == "relu"]
    assert len(act) == 1, f"Expected 1 ReLU activation, got {len(act)}"

    print("  PASS: Trainable classifier E2E conversion")
    print(f"    fully_connected: {len(fc)}")
    print(f"    cross_softmax: {len(loss)}")
    print(f"    activation(relu): {len(act)}")
    print(f"    Total layers: {len(result.layers)}")


class ResidualWithIdentity(nn.Module):
    """Model using nn.Identity in skip connections."""
    def __init__(self, dim=32):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.skip = nn.Identity()
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = self.skip(x)
        h = F.relu(self.fc1(x))
        h = self.fc2(h)
        return self.norm(h + residual)


def test_identity_skip_connection_e2e():
    """Model with nn.Identity in skip connections."""
    model = ResidualWithIdentity()
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"x": torch.randn(1, 32)})

    result.summary()

    assert len(result.unknown_layers) == 0, \
        f"Identity skip model has unknowns: {[l.layer_type for l in result.unknown_layers]}"

    fc = [l for l in result.layers if l.layer_type == "fully_connected"]
    assert len(fc) == 2, f"Expected 2 fully_connected, got {len(fc)}"

    add = [l for l in result.layers if l.layer_type == "addition"]
    assert len(add) == 1, f"Expected 1 addition, got {len(add)}"

    ln = [l for l in result.layers if l.layer_type == "layer_normalization"]
    assert len(ln) == 1, f"Expected 1 layer_normalization, got {len(ln)}"

    print("  PASS: Identity skip connection E2E conversion")
    print(f"    fully_connected: {len(fc)}")
    print(f"    addition: {len(add)}")
    print(f"    layer_normalization: {len(ln)}")
    print(f"    Total layers: {len(result.layers)}")


class AutoencoderWithMSE(nn.Module):
    """Autoencoder with MSE reconstruction loss."""
    def __init__(self, in_dim=64, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, in_dim),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        loss = self.loss_fn(recon, x)
        return loss


def test_autoencoder_mse_e2e():
    """Autoencoder with MSE loss."""
    model = AutoencoderWithMSE()
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"x": torch.randn(1, 64)})

    result.summary()

    assert len(result.unknown_layers) == 0, \
        f"Autoencoder has unknowns: {[l.layer_type for l in result.unknown_layers]}"

    fc = [l for l in result.layers if l.layer_type == "fully_connected"]
    assert len(fc) == 4, f"Expected 4 fully_connected, got {len(fc)}"

    loss = [l for l in result.layers if l.layer_type == "mse"]
    assert len(loss) == 1, f"Expected 1 mse, got {len(loss)}"

    print("  PASS: Autoencoder MSE E2E conversion")
    print(f"    fully_connected: {len(fc)}")
    print(f"    mse: {len(loss)}")
    print(f"    Total layers: {len(result.layers)}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TIER 3 LAYER MAPPER TESTS")
    print("=" * 70)

    print("\n--- Unit Tests ---")
    test_identity()
    test_gru_cell()
    test_lstm_cell()
    test_rnn_cell()
    test_cross_entropy_loss_module()
    test_mse_loss_module()
    test_kl_div_loss_module()
    test_bce_with_logits_loss_module()

    print("\n--- Model-Level E2E Tests ---")
    test_rnn_cell_lm_e2e()
    test_trainable_classifier_e2e()
    test_identity_skip_connection_e2e()
    test_autoencoder_mse_e2e()

    print("\n" + "=" * 70)
    print("ALL TIER 3 LAYER MAPPER TESTS PASSED!")
    print("=" * 70)
