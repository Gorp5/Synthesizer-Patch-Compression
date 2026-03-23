import math

import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange

class SpectrogramRefiner(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(1, 32, 3, padding=1),
            nn.GELU(),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU(),

            nn.Conv2d(32, 16, 3, padding=1),
            nn.GELU(),

            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, x):

        x = x.unsqueeze(1)

        residual = self.net(x)

        return (x + residual).squeeze(1)

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, num_positions: int, dim: int):
        super().__init__()
        pe = torch.zeros(num_positions, dim)
        position = torch.arange(0, num_positions).unsqueeze(1)
        div_term = torch.exp( torch.arange(0, dim, 2) * (-math.log(10000.0) / dim) )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, B: int):
        return self.pe.unsqueeze(0).expand(B, -1, -1)

class SpectrogramPositionalEmbedding(nn.Module):
    def __init__(self, freq_patches, time_patches, dim):
        super().__init__()

        self.freq_embed = SinusoidalPositionalEmbedding(freq_patches, dim // 2)
        self.time_embed = SinusoidalPositionalEmbedding(time_patches, dim // 2)

        self.freq_patches = freq_patches
        self.time_patches = time_patches

    def forward(self, B):

        freq = self.freq_embed.pe
        time = self.time_embed.pe

        freq = freq[:, None, :]
        time = time[None, :, :]

        pos = torch.cat([
            freq.expand(-1, self.time_patches, -1),
            time.expand(self.freq_patches, -1, -1)
        ], dim=-1)

        pos = pos.view(self.freq_patches * self.time_patches, -1)

        return pos.unsqueeze(0).expand(B, -1, -1)


class MetricBiasUpdater(nn.Module):

    def __init__(
        self,
        d_model: int,
        geom_dim: int = 32,
        alpha: float = 1.0,
        beta: float = 0.05,
        clamp_value: float = 10.0,
        learnable_scales: bool = True,
    ):
        super().__init__()

        # Projection into geometry space
        self.geom_proj = nn.Linear(d_model, geom_dim, bias=False)

        # Scale parameters
        if learnable_scales:
            self.alpha = nn.Parameter(torch.tensor(alpha))
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer("alpha", torch.tensor(alpha))
            self.register_buffer("beta", torch.tensor(beta))

        self.clamp_value = clamp_value

        # Initialize near zero so early training ≈ no geometry modulation
        nn.init.normal_(self.geom_proj.weight, mean=0.0, std=1e-3)

    def forward(self, H, B_prev):
        G = self.geom_proj(H)

        # ||x - y||^2 = x^2 + y^2 - 2xy
        G_sq = (G ** 2).sum(dim=-1, keepdim=True)
        dist = G_sq + G_sq.transpose(1, 2) - 2 * torch.matmul(G, G.transpose(1, 2))
        dist = torch.clamp(dist, min=0.0)  # numerical safety

        delta_B = -dist

        B_next = self.alpha * B_prev + self.beta * delta_B

        B_next = torch.clamp(B_next, -self.clamp_value, self.clamp_value)

        return B_next


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()

        dim_head = dim / heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, alibi_bias=None):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        if alibi_bias is not None:
            alibi_bias = alibi_bias.unsqueeze(1)
        else:
            alibi_bias = None

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=alibi_bias,  # flash attention supports additive mask as attn_mask
        )

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads),
                FeedForward(dim, mlp_dim),
                MetricBiasUpdater(d_model=dim, geom_dim=32, learnable_scales=True)
            ]))

    def forward(self, x, algorithm_distance_matricies=None):
        alibi_bias = algorithm_distance_matricies

        for attn, ff, alibi_bias_function in self.layers:

            attn_out = attn(
                x,
                alibi_bias,
            )

            x = attn_out + x

            x = ff(x) + x

            alibi_bias = alibi_bias_function(x, alibi_bias)

        x = self.norm(x)

        return x, alibi_bias

class CrossTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CrossAttention(dim, heads),
                Attention(dim, heads=heads),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x, cross_tokens, conditioning_token_mask=None):
        for cross, attn, ff in self.layers:
            x = x + cross(
                x,
                cross_tokens,
                conditioning_token_mask
            )

            attn_out = attn(
                x,
            )

            x = attn_out + x
            x = ff(x) + x

        x = self.norm(x)

        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        dim_head = dim // heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

        self.attend = nn.Softmax(dim=-1)

    def forward(self, x, target):
        q = self.to_q(self.norm_q(x))
        k, v = self.to_kv(self.norm_kv(target)).chunk(2, dim=-1)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
        )

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class GraphTransformerMidi(nn.Module):
    def __init__(
            self,
            input_size,
            d_model,
            depth,
            heads,
            mlp_dim,
            num_global_params=24,
            algorithm_distance_matricies=False,
            device="cuda",
    ):
        super().__init__()

        self.device = device
        self.input_size = input_size
        self.num_nodes = 6
        self.algorithm_index = 176
        self.d_model = d_model

        self.algorithm_distance_matricies = algorithm_distance_matricies.to("cuda")

        self.input_projection = nn.Linear(input_size, d_model)
        self.encoder = Transformer(d_model, depth, heads, mlp_dim)

        self.enc_global_film_gamma = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU())
        self.enc_global_film_beta = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU())

        self.global_embedding = nn.Sequential(
            nn.Linear(num_global_params, d_model),
        )

        self.global_pred = nn.Sequential(
            nn.Linear(d_model, num_global_params)
        )

        self.patch_node_pos = nn.Parameter(torch.randn(7, d_model))
        self.midi_pos_emb = nn.Parameter(torch.randn(12, d_model))
        self.patch_pos_emb = nn.Parameter(torch.randn(7, d_model))
        self.graph_pos_emb = nn.Parameter(torch.randn(1, d_model))

        self.spectrogram_decoder = CrossTransformer(d_model, depth, heads, mlp_dim)

        self.patch_size = 16
        self.height_in_patches = 128 // self.patch_size
        self. width_in_patches = 256 // self.patch_size

        total_patches = self.height_in_patches * self.width_in_patches

        self.patch_queries = nn.Parameter(torch.randn(total_patches, d_model))
        nn.init.normal_(self.patch_queries, std=0.02)

        self.patch_positional_embeddings = SpectrogramPositionalEmbedding(self.height_in_patches, self.width_in_patches, d_model)

        self.to_spectrogram = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, self.patch_size * self.patch_size),
            Rearrange(
                'b (f t) (pf pt) -> b (f pf) (t pt)',
                f=self.height_in_patches,
                t=self.width_in_patches,
                pf=self.patch_size,
                pt=self.patch_size
            )
        )

        self.midi_proj = nn.Linear(4, d_model)
        self.bias_proj = nn.Linear(49, d_model)
        self.timbre_proj = nn.Linear(d_model, d_model)

        self.timbre_norm = nn.LayerNorm(d_model)
        self.midi_norm = nn.LayerNorm(d_model)
        self.bias_norm = nn.LayerNorm(d_model)
        self.query_norm = nn.LayerNorm(d_model)

        self.start_token = nn.Parameter(torch.randn(d_model))
        self.register_buffer(
            "velocity_bins",
            torch.tensor([37, 75, 113])
        )


    def midi_to_token_simple(self, notes):
        start_offset = 0
        duration_offset = 512
        pitch_offset = 1024
        velocity_offset = 1536

        B, N, _ = notes.shape

        vel_tokens = torch.bucketize(notes[:, :, 3].contiguous(), self.velocity_bins)
        duration = notes[:, :, 1] - notes[:, :, 0]

        return torch.stack([
            notes[:, :, 0] + start_offset,
            duration + duration_offset,
            notes[:, :, 2] + pitch_offset,
            vel_tokens + velocity_offset], dim=2).to(torch.float)

    def forward(
            self, x, midi
    ):
        node_param_dim = self.input_size
        num_nodes = 6
        node_end = num_nodes * node_param_dim
        node_flat = x[:, :node_end]
        algorithm_id = x[:, self.algorithm_index:self.algorithm_index + 32].to(torch.int)
        global_params_half_one = x[:, node_end:self.algorithm_index]
        global_params_half_two = x[:, self.algorithm_index + 32:]
        global_params = torch.cat((global_params_half_one, global_params_half_two), dim=1)

        if self.algorithm_distance_matricies is not None:
            algorithm_distance_matricies = self.algorithm_distance_matricies[torch.argmax(algorithm_id, dim=1)].to(torch.float)
        else:
            algorithm_distance_matricies = None

        B, F = node_flat.shape
        nodes = node_flat.view(B, num_nodes, node_param_dim)
        x_p = self.input_projection(nodes)
        global_emb_enc = self.global_embedding(global_params).unsqueeze(1)
        x_p = torch.cat([global_emb_enc, x_p], dim=1)
        x_p = x_p + self.patch_node_pos.unsqueeze(0)

        x_p, biases = self.encoder(x_p, algorithm_distance_matricies=algorithm_distance_matricies)

        midi_tokens = self.midi_to_token_simple(midi)
        mB, mT, mF = midi_tokens.shape

        midi_tokens = self.midi_norm(self.midi_proj(midi_tokens))
        x_p = self.timbre_norm(self.timbre_proj(x_p))

        biases = self.bias_proj(biases.view(B, -1)).unsqueeze(1)

        midi_tokens = midi_tokens   + self.midi_pos_emb.unsqueeze(0)[:, :mT, :]
        patch_tokens = x_p          + self.patch_pos_emb.unsqueeze(0)
        graph_token = biases       + self.graph_pos_emb.unsqueeze(0)

        conditioning_tokens = torch.cat([midi_tokens, graph_token, patch_tokens], dim=1)

        queries = self.patch_queries.unsqueeze(0).expand(B, -1, -1)
        queries = self.query_norm(queries)
        queries = queries + self.patch_positional_embeddings(B)

        latent_tokens = self.spectrogram_decoder(
            queries,
            conditioning_tokens,
        )

        # patch_pixels = self.patch_head(latent_tokens)
        #
        # coarse = patch_pixels.view(
        #     B,
        #     self.height_in_patches,
        #     self.width_in_patches,
        #     self.patch_size,
        #     self.patch_size
        # )
        #
        # coarse = coarse.permute(0, 1, 3, 2, 4).reshape(
        #     B,
        #     self.height_in_patches * self.patch_size,
        #     self.width_in_patches * self.patch_size
        # )
        #
        # recon = self.refiner(coarse)

        recon = self.to_spectrogram(latent_tokens)
        return recon
