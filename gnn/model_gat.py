import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class BN1d(nn.Module):
    def __init__(self, out_dim, use_bn):
        super(BN1d, self).__init__()
        self.use_bn = use_bn
        self.bn = nn.BatchNorm1d(out_dim)
             
    def forward(self, x):
        if not self.use_bn:
            return  x
        origin_shape = x.shape
        x = x.view(-1, origin_shape[-1])
        x = self.bn(x)
        x = x.view(origin_shape)
        return x

class GAT(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=1, dropout=0.6, alpha=0.2):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.num_heads = num_heads
        self.output_dim = output_dim // num_heads

        # 모든 헤드에 대한 단일 W와 a
        self.W = nn.Parameter(torch.empty(size=(input_dim, num_heads * self.output_dim)))
        self.a = nn.Parameter(torch.empty(size=(2 * self.output_dim, 1)))

        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, X, A):
        # X: [batch_size, num_nodes, input_dim]
        # A: [batch_size, num_nodes, num_nodes]
        B, N, _ = X.size()

        # 1. 선형 변환: [batch_size, num_nodes, num_heads * output_dim]
        H = torch.mm(X.view(-1, X.size(-1)), self.W).view(B, N, self.num_heads, -1)
        print(f'H: {H.size()}')

        # 2. 어텐션 계수 계산 준비
        a_input = self._prepare_attentional_mechanism_input(H)
        print(f'a_input: {a_input.size()}')

        # 3. 어텐션 계수 계산: e = LeakyReLU(a^T [Wh_i || Wh_j])
        e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(-1)
        print(f'e: {e.size()}')

        # 4. 마스킹 및 정규화: α = softmax(e)
        attention = self._get_attention_weights(e, A)
        print(f'attention: {attention.size()}')

        # 5. 최종 노드 표현 계산: h' = σ(Σ α * Wh)
        H_prime = torch.zeros(B, N, self.num_heads, self.num_heads * self.output_dim, device=X.device)
        # for i in range(N):
        #     for j in range(self.num_heads):
        #         H_prime[i, j] = torch.mm(attention[i, j].unsqueeze(0), H[:, j, :])
        for b in range(B):
            for i in range(N):
                for j in range(self.num_heads):
                    H_prime[b, i, j] = torch.mm(attention[b, i, j].unsqueeze(0), H[b, :, j, :])

        print(f'H_prime: {H_prime.size()}')

        # 6. 모든 헤드의 결과를 연결
        H_multi = H_prime.view(B, N, -1)
        print(f'H_multi: {H_multi.size()}')

        return H_multi, A
        
    #     return torch.cat([H_expanded, H_expanded_t], dim=-1)
    def _prepare_attentional_mechanism_input(self, H):
        B, N, num_heads, out_dim = H.size()  # B: 배치사이즈, N: 노드 개수, out_dim: 특성 차원
        
        # H를 [B, N, 1, N, out_dim]로 확장
        H_expanded = H.repeat(1, 1, 1, N).view(B, N, num_heads, N, out_dim)  # (B, N, num_heads, N, out_dim)
        
        # H를 [B, N, 1, N, out_dim]로 확장
        H_expanded_t = H.repeat(1, 1, N, 1).view(B, N, num_heads, N, out_dim)  # (B, N, num_heads, N, out_dim)
        
        # 두 텐서를 마지막 차원에서 결합하여 [B, N, N, out_dim]을 얻음
        return torch.cat([H_expanded, H_expanded_t], dim=-1)  # (B, N, N, out_dim)



    def _get_attention_weights(self, e, A):
        # e: [batch_size, num_nodes, num_nodes, num_heads]
        # A: [batch_size, num_nodes, num_nodes]

        # 마스킹: 연결되지 않은 노드 쌍의 어텐션 계수를 매우 작은 음수로 설정
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(A.unsqueeze(2).expand_as(e) > 0, e, zero_vec)
        
        # 정규화: softmax 적용
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        return attention

class GATLayer(nn.Module):
    def __init__(self, input_dim, output_dim, use_bn, num_heads=1, dropout=0.6, alpha=0.2):
        super(GATLayer, self).__init__()
        self.gat = GAT(input_dim, output_dim, num_heads, dropout, alpha)
        self.bn = BN1d(output_dim, use_bn)

    def forward(self, X, A):
        H, A = self.gat(X, A)
        H = self.bn(H)
        H = F.elu(H)
        return H, A
    
class Readout(nn.Module):
    def __init__(self, out_dim, molvec_dim):
        super(Readout, self).__init__()
        self.readout_fc = nn.Linear(out_dim, molvec_dim)
        nn.init.xavier_normal_(self.readout_fc.weight.data)

    def forward(self, output_H):
        molvec = self.readout_fc(output_H)
        molvec = torch.mean(molvec, dim=1)
        return molvec
    

class GATNet(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        
        # Create Atom Element embedding layer
        self.embedding = self.create_emb_layer([args.vocab_size, args.degree_size,
                                                args.numH_size, args.valence_size,
                                                args.isarom_size],  args.emb_train)    
        
        self.gat_layers = nn.ModuleList()
        for i in range(args.n_layer):
            self.gat_layers.append(GATLayer(args.in_dim if i==0 else args.out_dim, args.out_dim, args.use_bn))
                                   
        self.readout = Readout(args.out_dim, args.molvec_dim)
        
        self.fc1 = nn.Linear(args.molvec_dim, args.molvec_dim//2)
        self.fc2 = nn.Linear(args.molvec_dim//2, args.molvec_dim//2)
        self.fc3 = nn.Linear(args.molvec_dim//2, 1)
        self.relu = nn.ReLU()
        
    def create_emb_layer(self, list_vocab_size, emb_train=False):
        list_emb_layer = nn.ModuleList()
        for i, vocab_size in enumerate(list_vocab_size):
            vocab_size += 1
            emb_layer = nn.Embedding(vocab_size, vocab_size)
            weight_matrix = torch.zeros((vocab_size, vocab_size))
            for i in range(vocab_size):
                weight_matrix[i][i] = 1
            emb_layer.load_state_dict({'weight': weight_matrix})
            emb_layer.weight.requires_grad = emb_train
            list_emb_layer.append(emb_layer)
        return list_emb_layer

    def _embed(self, x):
        list_embed = list()
        for i in range(5):
            list_embed.append(self.embedding[i](x[:, :, i]))
        x = torch.cat(list_embed, 2)
        return x
        
    def forward(self, x, A):
        A = A.float()
        x = self._embed(x)   
        
        for i, module in enumerate(self.gat_layers):
            x, A = module(x, A)
        x = self.readout(x)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.squeeze(x)
        