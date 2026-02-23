from einops import rearrange, repeat
import torch
import torch.nn as nn


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
    def __init__(self, dim, heads=4, sparsity_weight=0.0):
        super().__init__()

        dim_head = dim / heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

        # sparsity
        self.sparsity_weight = sparsity_weight

    def forward(self, x, alibi_bias=None, alg_film=None, return_attn_loss=False):
        x = self.norm(x)

        if alg_film is not None:
            gamma, beta = alg_film
            x = gamma[:, None, :] * x + beta[:, None, :]

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if alibi_bias is not None:
            alibi_bias = alibi_bias.unsqueeze(1).expand(-1, 4, -1, -1)
            dots = dots + alibi_bias

        attn = self.attend(dots)

        sparsity_loss = None
        if return_attn_loss and self.sparsity_weight > 0.0:
            eps = 1e-8
            entropy = -(attn * (attn + eps).log()).sum(dim=-1)
            sparsity_loss = entropy.mean() * self.sparsity_weight

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), sparsity_loss


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, sparsity_weight=0.0):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, sparsity_weight=sparsity_weight),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x, alg_film=None, global_film=None, return_attn_loss=False, algorithm_distance_matricies=None):
        alibi_bias = algorithm_distance_matricies
        total_sparsity_loss = 0.0

        for attn, ff in self.layers:
            attn_out, sparsity_loss = attn(
                x,
                alibi_bias,
                alg_film=alg_film,
                return_attn_loss=return_attn_loss,
            )

            x = attn_out + x

            if global_film is not None:
                gamma, beta = global_film
                x = gamma[:, None, :] * x + beta[:, None, :]

            x = ff(x) + x

            if sparsity_loss is not None:
                total_sparsity_loss = total_sparsity_loss + sparsity_loss

        x = self.norm(x)

        if return_attn_loss:
            return x, total_sparsity_loss

        return x


class GraphTransformerAutoencoderAlibi(nn.Module):
    def __init__(
            self,
            input_size,
            latent_space,
            d_model,
            depth,
            heads,
            mlp_dim,
            num_algorithms=32,
            num_global_params=24,
            sparsity_weight=0.0,
            reparameterization=False,
            algorithm_distance_matricies=False,
            device="cuda"
    ):
        super().__init__()

        self.device = device
        self.input_size = input_size
        self.input_projection = nn.Linear(input_size, d_model)
        self.encoder = Transformer(d_model, depth, heads, mlp_dim, sparsity_weight=sparsity_weight)

        self.reparameterization = reparameterization

        self.to_latent = nn.Linear(d_model, latent_space)
        if reparameterization:
            self.extra_logvar = nn.Linear(d_model, latent_space)

        self.from_latent = nn.Linear(latent_space, d_model)
        self.decoder = Transformer(d_model, depth, heads, mlp_dim, sparsity_weight=sparsity_weight)
        self.output_projection = nn.Linear(d_model, input_size)

        self.enc_global_film_gamma = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU())
        self.enc_global_film_beta = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU())

        self.dec_global_film_gamma = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU())
        self.dec_global_film_beta = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU())

        self.alg_pred = nn.Linear(latent_space, num_algorithms)

        self.global_embedding = nn.Sequential(
            nn.Linear(num_global_params, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        self.global_pred = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_global_params)
        )

        self.learned_embeddings = nn.Parameter(torch.randn(7, d_model))

        if algorithm_distance_matricies is not None:
            self.algorithm_distance_matricies = algorithm_distance_matricies.to("cuda")
        else:
            self.algorithm_distance_matricies = None

        self.loss = nn.CrossEntropyLoss(reduction="mean")
        self.num_nodes = 6


    def reparameterize(self, mean, logvar):
        epsilon = torch.randn_like(logvar).to(self.device)
        # return mean + var * epsilon
        return mean + torch.exp(logvar / 2) * epsilon

    def generate(self, latent):
        B = 1
        F = 225

        alg_probs = self.alg_pred(latent)

        algorithm_distance_matricies = self.algorithm_distance_matricies[torch.argmax(alg_probs, dim=1)]

        x_p = self.from_latent(latent)
        x_p = x_p.unsqueeze(1).repeat(1, self.num_nodes + 1, 1)

        x_p += self.learned_embeddings.expand(B, x_p.shape[1], x_p.shape[2])
        x_p = self.decoder(x_p, algorithm_distance_matricies=algorithm_distance_matricies)

        global_hat = self.global_pred(x_p[:, 0])
        x_p = self.output_projection(x_p[:, 1:])
        x_p = x_p.view(B, F)

        new_algo_index = 8
        x_recon = torch.cat([
            x_p,
            global_hat[:, new_algo_index:],
            alg_probs,
            global_hat[:, :new_algo_index]
        ], dim=1)

        return x_recon

    def forward(self, x, return_attn_loss=True):
        node_param_dim = self.input_size
        num_nodes = 6

        if x.shape[1] != 225:
            print("uh")

        node_end = num_nodes * node_param_dim
        algorithm_index = 176

        node_flat = x[:, :node_end]
        algorithm_id = x[:, algorithm_index:algorithm_index + 32].to(torch.int)
        global_params_half_one = x[:, node_end:algorithm_index]
        global_params_half_two = x[:, algorithm_index + 32:]
        global_params = torch.cat((global_params_half_one, global_params_half_two), dim=1)

        if self.algorithm_distance_matricies is not None:
            algorithm_distance_matricies = self.algorithm_distance_matricies[torch.argmax(algorithm_id, dim=1)]
        else:
            algorithm_distance_matricies = None

        B, F = node_flat.shape

        nodes = node_flat.view(
            B,
            num_nodes,
            node_param_dim
        )

        x_p = self.input_projection(nodes)

        global_emb_enc = self.global_embedding(global_params).unsqueeze(1)

        x_p = torch.cat([global_emb_enc, x_p], dim=1)

        if return_attn_loss:
            x_p += self.learned_embeddings.expand(B, x_p.shape[1], x_p.shape[2])
            x_p, sparsity_loss_enc = self.encoder(
                x_p, return_attn_loss=True, algorithm_distance_matricies=algorithm_distance_matricies
            )
        else:
            x_p += self.learned_embeddings.expand(B, x_p.shape[1], x_p.shape[2])
            x_p = self.encoder(x_p, algorithm_distance_matricies=algorithm_distance_matricies)

        # Pool over nodes to get graph-level latent
        x_p = x_p.mean(dim=1)
        latent = self.to_latent(x_p)

        logvar = torch.ones((1), device=self.device)
        if self.reparameterization:
            logvar = self.extra_logvar(x_p)
            latent = self.reparameterize(latent, logvar)

        alg_probs = self.alg_pred(latent)

        algorithm_distance_matricies = self.algorithm_distance_matricies[torch.argmax(alg_probs, dim=1)]

        x_p = self.from_latent(latent)
        x_p = x_p.unsqueeze(1).repeat(1, num_nodes + 1, 1)

        if return_attn_loss:
            x_p += self.learned_embeddings.expand(B, x_p.shape[1], x_p.shape[2])
            x_p, sparsity_loss_dec = self.decoder(
                x_p, return_attn_loss=True, algorithm_distance_matricies=algorithm_distance_matricies
            )
        else:
            x_p += self.learned_embeddings.expand(B, x_p.shape[1], x_p.shape[2])
            x_p = self.decoder(x_p, algorithm_distance_matricies=algorithm_distance_matricies)

        global_hat = self.global_pred(x_p[:, 0])
        x_p = self.output_projection(x_p[:, 1:])
        x_p = x_p.view(B, F)

        new_algo_index = 8
        x_recon = torch.cat([
            x_p,
            global_hat[:, new_algo_index:],
            alg_probs,
            global_hat[:, :new_algo_index]
        ], dim=1)

        return x_recon, latent, logvar, (alg_probs)