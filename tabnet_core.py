# tabnet_core.py — Pure TabNet definition (safe to import)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- sparsemax / entmax ----
class SparsemaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim=-1):
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input = input - max_val
        sorted_input, _ = torch.sort(input, descending=True, dim=dim)
        input_cumsum = sorted_input.cumsum(dim) - 1
        rhos = torch.arange(1, sorted_input.size(dim)+1,
                            device=sorted_input.device, dtype=sorted_input.dtype)
        view = [1]*input.dim(); view[dim] = -1
        rhos = rhos.view(view)
        support = (rhos * sorted_input) > input_cumsum
        support_size = support.sum(dim=dim, keepdim=True)
        tau = input_cumsum.gather(dim, support_size - 1) / support_size.to(input.dtype)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(support_size, output)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        support_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0
        v_hat = grad_input.sum(dim=dim, keepdim=True) / support_size.to(output.dtype)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None

def sparsemax(input, dim=-1): return SparsemaxFunction.apply(input, dim)

class Sparsemax(nn.Module):
    def __init__(self, dim=-1): super().__init__(); self.dim=dim
    def forward(self, x): return sparsemax(x, self.dim)

class Entmax15Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim=-1):
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        X = (input - max_val) / 2
        Xsrt, _ = torch.sort(X, descending=True, dim=dim)
        rho = torch.arange(1, Xsrt.size(dim)+1,
                           device=X.device, dtype=X.dtype)
        view = [1]*X.dim(); view[dim] = -1
        rho = rho.view(view)
        csum = Xsrt.cumsum(dim)
        mean = csum / rho
        mean_sq = (Xsrt**2).cumsum(dim) / rho
        ss = rho * (mean_sq - mean**2)
        delta = torch.clamp((1 - ss) / rho, min=0)
        tau = mean - torch.sqrt(delta)
        support = (tau <= Xsrt).sum(dim=dim, keepdim=True)
        tau_star = tau.gather(dim, support - 1)
        out = torch.clamp(X - tau_star, min=0)**2
        ctx.save_for_backward(out)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        (Y,) = ctx.saved_tensors; dim = ctx.dim
        gppr = Y.sqrt()
        dX = grad_output * gppr
        q = dX.sum(dim=dim, keepdim=True) / gppr.sum(dim=dim, keepdim=True)
        dX -= q * gppr
        return dX, None

def entmax15(input, dim=-1): return Entmax15Function.apply(input, dim)

class Entmax15(nn.Module):
    def __init__(self, dim=-1): super().__init__(); self.dim=dim
    def forward(self, x): return entmax15(x, self.dim)

# ---- init utils ----
def initialize_non_glu(m, i, o): nn.init.xavier_normal_(m.weight, gain=np.sqrt((i+o)/np.sqrt(4*i)))
def initialize_glu(m, i, o):     nn.init.xavier_normal_(m.weight, gain=np.sqrt((i+o)/np.sqrt(i)))

# ---- blocks ----
class GBN(nn.Module):
    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):
        super().__init__(); self.bn = nn.BatchNorm1d(input_dim, momentum=momentum); self.vbs = virtual_batch_size
    def forward(self, x):
        if x.size(0) <= self.vbs: return self.bn(x)
        chunks = x.chunk(int(np.ceil(x.size(0)/self.vbs)), dim=0)
        return torch.cat([self.bn(c) for c in chunks], dim=0)

class GLU_Layer(nn.Module):
    def __init__(self, input_dim, output_dim, fc=None, virtual_batch_size=128, momentum=0.02):
        super().__init__(); self.output_dim = output_dim
        self.fc = fc or nn.Linear(input_dim, 2*output_dim, bias=False)
        initialize_glu(self.fc, input_dim, 2*output_dim)
        self.bn = GBN(2*output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum)
    def forward(self, x):
        x = self.bn(self.fc(x))
        return x[:, :self.output_dim] * torch.sigmoid(x[:, self.output_dim:])

class GLU_Block(nn.Module):
    def __init__(self, input_dim, output_dim, n_glu=2, first=False, shared_layers=None, virtual_batch_size=128, momentum=0.02):
        super().__init__(); self.first = first; self.glu_layers = nn.ModuleList()
        in_dim = input_dim
        if shared_layers is not None:
            for i in range(n_glu):
                fc = shared_layers[i] if i < len(shared_layers) else None
                self.glu_layers.append(GLU_Layer(in_dim if (i==0 and first) else output_dim,
                                                 output_dim, fc=fc, virtual_batch_size=virtual_batch_size, momentum=momentum))
        else:
            for i in range(n_glu):
                self.glu_layers.append(GLU_Layer(in_dim, output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum))
                in_dim = output_dim
    def forward(self, x):
        out = self.glu_layers[0](x)
        x = out if self.first else 0.5**0.5 * (out + x)
        for g in self.glu_layers[1:]:
            out = g(x)
            x = 0.5**0.5 * (out + x)
        return x

class FeatTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, shared_layers=None, n_glu_independent=2, virtual_batch_size=128, momentum=0.02):
        super().__init__()
        self.shared = GLU_Block(input_dim, output_dim, n_glu=len(shared_layers), first=True, shared_layers=shared_layers,
                                virtual_batch_size=virtual_batch_size, momentum=momentum) if shared_layers else nn.Identity()
        is_first = shared_layers is None
        self.specifics = GLU_Block(input_dim if is_first else output_dim, output_dim,
                                   n_glu=n_glu_independent, first=not is_first, shared_layers=None,
                                   virtual_batch_size=virtual_batch_size, momentum=momentum) if n_glu_independent>0 else nn.Identity()
    def forward(self, x): return self.specifics(self.shared(x))

class AttentiveTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, group_matrix=None, virtual_batch_size=128, momentum=0.02, mask_type="sparsemax"):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False); initialize_non_glu(self.fc, input_dim, output_dim)
        self.bn = GBN(output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum)
        self.selector = Sparsemax(dim=-1) if mask_type=="sparsemax" else Entmax15(dim=-1)
        # ★ register buffer so it moves with .to(device)
        if group_matrix is not None:
            self.register_buffer("group_matrix", group_matrix.detach().clone().to(torch.float32))
        else:
            self.group_matrix = None
    def forward(self, prior, x):
        m = self.selector(self.bn(self.fc(x)) * prior)
        return m

class EmbeddingGenerator(nn.Module):
    def __init__(self, input_dim, cat_dims=None, cat_idxs=None, cat_emb_dim=1, group_matrix=None):
        super().__init__()
        cat_dims, cat_idxs = cat_dims or [], cat_idxs or []
        self.skip_embedding = (len(cat_dims)==0 or len(cat_idxs)==0)
        if self.skip_embedding:
            self.post_embed_dim = input_dim
            # ★ register buffer here as well
            if group_matrix is not None:
                self.register_buffer("embedding_group_matrix", group_matrix.detach().clone().to(torch.float32))
            else:
                self.register_buffer("embedding_group_matrix", torch.eye(input_dim, dtype=torch.float32))
            return
        # categorical embeddings
        if isinstance(cat_emb_dim, int): cat_emb_dims = [cat_emb_dim]*len(cat_idxs)
        else: cat_emb_dims = cat_emb_dim
        self.post_embed_dim = int(input_dim + sum(cat_emb_dims) - len(cat_emb_dims))
        self.embeddings = nn.ModuleList([nn.Embedding(cd, ed) for cd, ed in zip(cat_dims, cat_emb_dims)])
        continuous_mask = torch.ones(input_dim, dtype=torch.bool); continuous_mask[cat_idxs] = False
        if group_matrix is not None:
            n_groups = group_matrix.shape[0]
            emb_g = torch.empty((n_groups, self.post_embed_dim), dtype=torch.float32)
            emb_counter = 0; post = 0
            for feat_idx in range(input_dim):
                if continuous_mask[feat_idx]:
                    emb_g[:, post] = group_matrix[:, feat_idx]
                    post += 1
                else:
                    n_emb = cat_emb_dims[emb_counter]
                    emb_g[:, post:post+n_emb] = group_matrix[:, feat_idx][:, None] / n_emb
                    post += n_emb; emb_counter += 1
            # ★ register buffer
            self.register_buffer("embedding_group_matrix", emb_g)
        else:
            self.register_buffer("embedding_group_matrix", torch.eye(self.post_embed_dim, dtype=torch.float32))
        self.continuous_mask = continuous_mask
    def forward(self, x):
        if self.skip_embedding: return x.float()
        cols = []; cat_counter = 0
        for i in range(x.shape[1]):
            if self.continuous_mask[i]:
                cols.append(x[:, i:i+1].float())
            else:
                cols.append(self.embeddings[cat_counter](x[:, i].long()))
                cat_counter += 1
        return torch.cat(cols, dim=1)

class TabNetEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, n_d=8, n_a=8, n_steps=3, gamma=1.3,
                 n_independent=2, n_shared=2, epsilon=1e-15, virtual_batch_size=128,
                 momentum=0.02, mask_type="sparsemax", group_attention_matrix=None):
        super().__init__()
        self.n_d, self.n_a, self.n_steps, self.gamma, self.epsilon = n_d, n_a, n_steps, gamma, epsilon
        self.initial_bn = nn.BatchNorm1d(input_dim, momentum=0.01)
        # shared GLU fc layers
        self.shared_layers = None
        if n_shared>0:
            self.shared_layers = nn.ModuleList()
            for i in range(n_shared):
                fc = nn.Linear(input_dim if i==0 else (n_d+n_a), 2*(n_d+n_a), bias=False)
                initialize_glu(fc, fc.in_features, fc.out_features)
                self.shared_layers.append(fc)
        # initial splitter
        self.initial_splitter = FeatTransformer(input_dim, n_d+n_a, shared_layers=self.shared_layers,
                                                n_glu_independent=n_independent, virtual_batch_size=virtual_batch_size, momentum=momentum)
        # attentive + feat transformers
        self.feat_transformers = nn.ModuleList()
        self.att_transformers  = nn.ModuleList()
        att_input_dim = group_attention_matrix.shape[0] if group_attention_matrix is not None else input_dim
        for _ in range(n_steps):
            self.feat_transformers.append(FeatTransformer(input_dim, n_d+n_a, shared_layers=self.shared_layers,
                                                          n_glu_independent=n_independent, virtual_batch_size=virtual_batch_size, momentum=momentum))
            self.att_transformers.append(AttentiveTransformer(n_a, att_input_dim, group_matrix=group_attention_matrix,
                                                              virtual_batch_size=virtual_batch_size, momentum=momentum, mask_type=mask_type))
        # head
        if isinstance(output_dim, list):
            self.multi_task_mappings = nn.ModuleList([nn.Linear(n_d, out_d, bias=False) for out_d in output_dim])
            for lin, out_d in zip(self.multi_task_mappings, output_dim): initialize_non_glu(lin, n_d, out_d)
            self.final_mapping = None
        else:
            self.multi_task_mappings = None
            self.final_mapping = nn.Linear(n_d, output_dim, bias=False); initialize_non_glu(self.final_mapping, n_d, output_dim)

    def forward(self, x, prior=None):
        x = self.initial_bn(x)
        if prior is None: prior = torch.ones(x.shape[0], x.shape[1], device=x.device)
        M_loss = 0.0
        x_ft = self.initial_splitter(x)
        decision_out = torch.zeros(x.shape[0], self.n_d, device=x.device)
        for step in range(self.n_steps):
            M = self.att_transformers[step](prior, x_ft[:, self.n_d:])
            M_loss += (M * torch.log(M + self.epsilon)).sum(dim=1).mean()
            prior = prior * (self.gamma - M)
            # ★ align device/dtype before matmul
            if getattr(self.att_transformers[step], "group_matrix", None) is not None:
                gm = self.att_transformers[step].group_matrix
                if gm.device != M.device or gm.dtype != M.dtype:
                    gm = gm.to(device=M.device, dtype=M.dtype)
                M_feat = M @ gm
            else:
                M_feat = M
            x_masked = M_feat * x
            x_ft = self.feat_transformers[step](x_masked)
            d = F.relu(x_ft[:, :self.n_d])
            decision_out += d
        if self.multi_task_mappings is not None:
            outs = [m(decision_out) for m in self.multi_task_mappings]
            return outs, M_loss
        else:
            return self.final_mapping(decision_out), M_loss

class TabNetNoEmbeddings(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(); self.encoder = TabNetEncoder(input_dim=input_dim, output_dim=output_dim, **kwargs)
    def forward(self, x): return self.encoder(x)
    def forward_masks(self, x): return self.encoder.forward_masks(x)

class TabNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_d=8, n_a=8, n_steps=3, gamma=1.3,
                 cat_idxs=None, cat_dims=None, cat_emb_dim=1, n_independent=2, n_shared=2,
                 epsilon=1e-15, virtual_batch_size=128, momentum=0.02, mask_type="sparsemax"):
        super().__init__()
        cat_idxs, cat_dims = cat_idxs or [], cat_dims or []
        self.embedder = EmbeddingGenerator(input_dim, cat_dims, cat_idxs, cat_emb_dim)
        self.post_embed_dim = self.embedder.post_embed_dim
        self.tabnet = TabNetNoEmbeddings(input_dim=self.post_embed_dim, output_dim=output_dim,
                                         n_d=n_d, n_a=n_a, n_steps=n_steps, gamma=gamma,
                                         n_independent=n_independent, n_shared=n_shared,
                                         epsilon=epsilon, virtual_batch_size=virtual_batch_size,
                                         momentum=momentum, mask_type=mask_type,
                                         group_attention_matrix=self.embedder.embedding_group_matrix)
    def forward(self, x):
        x_emb = self.embedder(x)
        return self.tabnet(x_emb)
