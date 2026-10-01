import torch
import torch.nn as nn
import torch.nn.functional as F


vocab = {
    "<cls>":0,
    "<sep>":1,
    '<pad>':2,
    "信用卡":3,
    "丢":4,
    "了":5,
    "怎么办":6,
    "遗失":7,
    "后":8,
    "立即":9,
    "挂失":10,
    "年费":11,
    "收费":12,
    "标准":13,
}


def build_reranker_data(vocab):

    """
    Cross Encoder 输入：

    [CLS]
    Query
    [SEP]
    Document
    [SEP]   

    label:
    1 -> 相关
    0 -> 不相关
    """

    # 正样本
    positive_input_ids = torch.tensor(
        [
            [
                vocab["<cls>"],
                vocab["信用卡"],
                vocab["丢"],
                vocab["了"],
                vocab["怎么办"],
                vocab["<sep>"],
                vocab["遗失"],
                vocab["后"],
                vocab["立即"],
                vocab["挂失"],
                vocab["<sep>"]
            ]
        ]
    )


    # 负样本
    negative_input_ids = torch.tensor(
        [
            [
                vocab["<cls>"],
                vocab["信用卡"],
                vocab["丢"],
                vocab["了"],
                vocab["怎么办"],
                vocab["<sep>"],
                vocab["年费"],
                vocab["收费"],
                vocab["标准"],
                vocab["<sep>"],
                vocab['<pad>']
            ]
        ]
    )

    # (batch_size,seq_len)
    input_ids = torch.cat(
        [
            positive_input_ids,
            negative_input_ids
        ],
        dim=0
    )

    # (batch_size)
    labels = torch.tensor(
        [
            1,
            0
        ],
        dtype=torch.float
    )


    return input_ids, labels


# 2. Encoder Block
class EncoderBlock(nn.Module):

    def __init__(self,embedding_dim,num_heads,ffn_dim):

        super().__init__()

        self.layernorm1 = nn.LayerNorm(embedding_dim)

        self.layernorm2 = nn.LayerNorm(embedding_dim)

        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim,ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim,embedding_dim)
        )


    def forward(self,x):

        """
        Encoder Attention:

        不需要 causal mask

        每个token可以看到全部token
        """

        hidden = self.layernorm1(x)

        '''与decoder_only唯一的区别就是不需要causal mask'''
        attention_output,attention_weights = self.self_attention(hidden,hidden,hidden)

        x = x + attention_output # 依旧整合上下文，残差连接

        hidden = self.layernorm2(x)

        ffn_output = self.ffn(hidden)

        x = x + ffn_output # 依旧整理所学，残差连接

        return x # 依旧维度不变，(batch_size,seq_len,embedding_dim)


# 3. Encoder
# ==========================
class Encoder(nn.Module):

    def __init__(self,vocab_size,seq_len,embedding_dim=64,num_blocks=2):

        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size,embedding_dim)

        self.position_embedding = nn.Embedding(seq_len,embedding_dim)

        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(embedding_dim=embedding_dim,num_heads=4,ffn_dim=128)
                for _ in range(num_blocks)
            ]
        )

        self.layernorm = nn.LayerNorm(embedding_dim)


    def forward(self,input_ids):

        """
        input:

        (batch_size,seq_len)

        output:

        (batch_size,seq_len,embedding_dim)
        """

        batch_size,seq_len = input_ids.shape

        # (seq_len)
        positions = torch.arange(seq_len,device=input_ids.device)

        # (seq_len,embedding_dim)
        position_embedding = self.position_embedding(positions)

        token_embedding = self.token_embedding(input_ids)

        x = (token_embedding + position_embedding.unsqueeze(0))

        for block in self.encoder_blocks:
            x = block(x)

        x = self.layernorm(x) # 依旧最后出来标准化

        return x


# 4. Cross Encoder Reranker
class CrossEncoderReranker(nn.Module):

    def __init__(self,vocab_size,seq_len,embedding_dim=64):

        super().__init__()

        self.encoder = Encoder(vocab_size=vocab_size,seq_len=seq_len,embedding_dim=embedding_dim)

        self.linear = nn.Linear(embedding_dim,1) # 1：得到最后的预测Label


    def forward(self,input_ids):

        hidden_states = self.encoder(input_ids)

        """
        hidden_states(batch_size,seq_len,embedding_dim)

        取每一句(实则这个例子只有一句)CLS:
        cls_embedding(batch_size,embedding_dim)
        """

        cls_embedding = hidden_states[:,0,:] # 因为已经结合和上下文，此时cls代表的就是一句话

        # score(batch_size,1)
        score = self.linear(cls_embedding)

        return score.squeeze(-1) #(batch_size)


# 5. Loss
def loss_fn(scores,labels):

    """
    Reranker不是预测token

    而是预测：query-document是否相关

    所以：score -> label
    """

    loss = F.binary_cross_entropy_with_logits(scores,labels)

    return loss


# 6. Train
def main():

    torch.manual_seed(15)

    '''因为reranker它不是decoder-only那种生成token，需要shift去对齐计算loss
    这玩意直接一个全塞进去最后输出一个score'''
    input_ids,labels = build_reranker_data(vocab)

    model = CrossEncoderReranker(vocab_size=len(vocab),seq_len=input_ids.shape[1])

    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)

    model.train()

    for epoch in range(300):

        scores = model(input_ids)

        loss = loss_fn(scores,labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if epoch % 100 == 0:
            print(
                f"epoch={epoch+1},loss={loss.item():.4f}"
            )


    # 测试排序
    model.eval()

    with torch.no_grad():

        scores = model(input_ids)

        probabilities = torch.sigmoid(scores) # 它不是指数，而是sigmoid函数，把所有实数转化0-1

        print("scores:",probabilities)



if __name__ == "__main__":

    main()