import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import Model, layers
import pathlib
import collections
import numpy as np
import math
from distutils.version import LooseVersion

# 设置随机种子
tf.random.set_seed(998)
np.random.seed(55)
# tensorflow version >= 2.0
assert tf.__version__ >= LooseVersion('2.0')


# @tf.function 转换为静态计算图
@tf.function
def gelu(x):
    """
    高斯误差线性单元激活函数在最近的 Transformer 模型（谷歌的 BERT 和 OpenAI 的 GPT-2）中得到了应用。GELU 的论文来自 2016 年
    是某些函数（比如双曲正切函数 tanh）与近似数值的组合.
    https://zhuanlan.zhihu.com/p/98863801
    """
    return 0.5 * x * (1 + tf.math.tanh(tf.math.sqrt(2 / math.pi) * (x + 0.044175 * tf.math.pow(x, 3))))

@tf.function
def swish(x):
    """和 ReLU 一样，Swish 无上界有下界。与 ReLU 不同的是，Swish 是平滑且非单调的函数。
    事实上，Swish 的非单调特性把它与大多数常见的激活函数区别开来;
    Swish 的一阶导和二阶导如图 2 所示。输入低于 1.25 时，导数小于 1。Swish 的成功说明 ReLU 的梯度不变性（即 x > 0 时导数为 1）在现代架构中或许不再是独有的优势。
    事实上，实验证明在使用批量归一化（Ioffe & Szegedy, 2015）的情况下，我们能够训练出比 ReLU 网络更深层的 Swish 网络
    https://zhuanlan.zhihu.com/p/30332306
    """
    return x * tf.math.sigmoid(x)


class Args:
    """准备参数"""
    # 为了促进这些残留连接，模型中的所有子层以及嵌入层均产生尺寸为dm = 512的输出。
    n_ctx = 512
    # embed层输出维度
    n_embed = 768
    n_head = 12
    # encoder:6层，decoder:6层
    n_layer = 12
    embed_pdrop = 0.1
    attn_pdrop = 0.1
    # ResidualDropout
    resid_pdrop = 0.1
    clf_pdrop = 0.1
    l2 = 0.1
    n_transfer = 12
    lm_coef = 0.5
    # the Adam optimizer with β1 = 0.9, β2 = 0.98 and eps = 10−9
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    n_valid = 374
    afn = gelu

# 设置权重初始化方法
zeros_init = keras.initializers.Zeros()
ones_init = keras.initializers.Ones()

class LayerNorm(Model):
    """构造一个OpenAI样式的layernorm模块（平方根内的epsilon）"""
    def __init__(self, n_state=768, epsilon=1e-5):
        super(LayerNorm, self).__init__()
        self.g = self.add_weight(shape=[n_state], initializer=ones_init)
        self.b = self.add_weight(shape=[n_state], initializer=zeros_init)
        self.epsilon = epsilon

    def call(self, inputs):
        # 求输入的均值
        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        # 求输入的方差
        std_2 = tf.reduce_mean(tf.math.pow(inputs - mean, 2), axis=-1, keepdims=True)
        # Norm
        out = (inputs - mean) / tf.math.sqrt(std_2 + self.epsilon)
        return self.g * out + self.b

class Conv1D(Model):

    def __init__(self, nf=768  * 3, rf=1, nx = 768):
        super(Conv1D, self).__init__()

        self.nf = nf
        self.rf = rf
        if rf == 1:  # 应用更快的一维卷积代替矩阵乘法运算
            self.w = self.add_weight(shape=[nx, nf], initializer=keras.initializers.RandomNormal(stddev=0.2))
            self.b = self.add_weight(shape=[nf], initializer=zeros_init)
        else:
            raise NotImplementedError

    def call(self, inputs):
        if self.rf == 1:
            size_out = list(inputs.shape[:-1]) + [self.nf]
            # 进行一维 Conv
            x = tf.reshape(inputs, [-1, inputs.shape[-1]]) @ self.w + self.b
            x = tf.reshape(x, size_out)
        else:
            raise NotImplementedError
        return x

class Attention(Model):
    """learning task-independent sentence representations [4, 27, 28, 22]. """

    def __init__(self, nx=768, n_ctx:"the embedding layers, produce outputs of dimension "=512,
                 cfg=Args, scale=False):
        super(Attention, self).__init__()
        n_state = nx
        assert n_state % cfg.n_head == 0
        self.b = self.add_weight(shape=[1, 1, n_ctx, n_ctx], initializer=keras.initializers.Zeros())
        self.b.assign(tf.linalg.LinearOperatorLowerTriangular().to_dense())
        self.n_head = cfg.n_head
        self.scale = scale
        # Linear
        self.c_attn = Conv1D(n_state * 3, 1, nx)
        self.c_proj = Conv1D(n_state, 1, nx)
        self.attn_dropout = layers.Dropout(cfg.attn_pdrop)
        self.resid_dropout = layers.Dropout(cfg.resid_pdrop)

    def call(self, inputs):
        # 对inputs进行一维卷积
        # Linear
        x = self.c_attn(inputs)
        query, key, value = tf.split(x, num_or_size_splits=3, axis=2)
        # Scaled Dot-Product Attention
        query = self.s_dp_attn(x, k=True)
        # Scaled Dot-Product Attention
        value = self.s_dp_attn(x)
        # self-attn
        a = self._attn(query, key, value)
        # Multi-Head Attention
        a = self.mh_attn(a)
        # Linear
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a

    def _attn(self, q, k, v):
        """
        dk = dv = dmodel/h = 64
        :param q: Queries
        :param k: Keys
        :param v: Values
        :return: ......
        """
        score = q @ k
        if self.scale:
            score = score / 8 * tf.math.sqrt(v.shape[-1], tf.float32)
        # self.b may be larger than w, so we need to crop it
        b = self.b[:, :, :score.shape[-2], :score.shape[-1]]
        score = score * b + 1e-9 * (1 - b)
        score = tf.math.softmax(score, axis=-1)
        return score @ v

    def mh_attn(self, x):
        x = tf.transpose(x, [0, 2, 3, 1])
        new_x_shape = list(x.shape[:-2]) + [x.shape[-1] * x.shape[-2]]
        return tf.reshape(x, new_x_shape)  # in openai implem: fct merge_states

    def s_dp_attn(self, x, k=False):
        new_x_shape = list(x.shape[:-1]) + [self.n_head, x.shape[-1]//self.n_head]
        x = tf.reshape(x, new_x_shape)  # in openai implem: fct split_states
        if k:
            return tf.transpose(x, [0, 2, 3, 1])
        else:
            return tf.transpose(x, [0, 2, 1, 3])

class FFT(Model):
    """FFN(x) = max(0,xW1 + b1)W2 + b2"""
    def __init__(self, n_state=3072, cfg=Args):
        super(FFT, self).__init__()

        nx = cfg.n_embed
        self.c_fc = Conv1D(n_state, 1, nx)
        self.c_proj = Conv1D(nx, 1, n_state)
        # gelu激活函数
        self.act = cfg.afn
        self.dropout = layers.Dropout(cfg.resid_pdrop)

    def call(self, inputs):
        h = self.act(self.c_fc(inputs))
        h2 = self.c_proj(h)
        return self.dropout(h2)

class Block(Model):

    def __init__(self, n_ctx=512, cfg=Args, scale=False):
        super(Block, self).__init__()

        nx = cfg.n_embed
        self.attn = Attention(nx, n_ctx, cfg, scale)
        self.ln_1 = LayerNorm(nx)
        self.fft = FFT(4 * nx, cfg)
        self.ln_2 = LayerNorm(nx)

    def call(self, inputs):
        a = self.attn(inputs)
        n = self.ln_1(a + inputs)
        m = self.fft(n)
        h = self.ln_2(n + m)

class TransformerModel(Model):
    """构建Transformer模型"""
    def __init__(self, cfg=Args, vocab=40558, n_ctx=512):
        super(TransformerModel, self).__init__()

        self.vocab = vocab
        self.embed = layers.Embedding(vocab, cfg.n_embed)
        # 构造输入embed的位置信息
        self.embed.build([1])
        self.drop = layers.Dropout(cfg.embed_pdrop)
        self.h = [Block(n_ctx, cfg, scale=False) for _ in range(cfg.n_layer)]

    def call(self, inputs):
        x = tf.reshape(inputs, [-1, inputs.shape[-2], inputs.shape[-1]])
        e = self.drop(self.embed(x))
        # 将位置信息添加到输入嵌入中
        h = tf.reduce_sum(e, 2)
        for block in self.h:
            h = block(h)
        return h

class LMHead(Model):
    """构建语言模型头为: Transformer Model"""
    def __init__(self, model, cfg=Args, trunc_and_shape=True):
        super(LMHead, self).__init__()

        self.n_embed = cfg.n_embed
        embed_shape = model.embed.weights[0].shape
        self.embed = model.embed.weights[0]
        self.decoder = lambda x:x @ tf.transpose(self.embed)
        self.trunc_and_shape = trunc_and_shape

    def call(self, inputs):
        # 截断语言模型预测(remove the last token)
        h_trunc = tf.reshape(inputs[:, :-1], [-1, self.n_embed]) if self.trunc_and_shape else inputs
        lm_logits = self.decoder(h_trunc)
        return lm_logits

class MultipleChoiceHead(Model):
    """Transformer 的分类模型头"""
    def __init__(self, clf_token=40480, cfg=Args):
        super(MultipleChoiceHead, self).__init__()

        self.n_embed = cfg.n_embed
        self.n_ctx = cfg.n_ctx
        self.clf_token = clf_token
        self.dropout = layers.Dropout(cfg.clf_pdrop, noise_shape=[1, 2, cfg.n_embed, 1])
        self.linear = layers.Dense(1, input_shape=[cfg.n_embed],
                                   kernel_initializer=keras.initializers.RandomNormal(stddev=0.2),
                                   bias_initializer=keras.initializers.RandomNormal(stddev=1))
        self.linear.build(cfg.n_embed)



    def call(self,hidden, inputs):
        # 分类预测
        clf_h = tf.reshape(hidden, [-1, self.n_embed])
        flat = tf.reshape(inputs[..., 0], [-1])
        clf_h = tf.boolean_mask(clf_h, tf.equal(flat, self.clf_token))
        clf_h = tf.reshape(clf_h, [-1, inputs.shape[1], self.n_embed, 1])
        clf_h = self.dropout(clf_h)
        clf_h = tf.reshape(clf_h, [-1, self.n_embed])
        clf_logits = self.linear(clf_h)

        return tf.reshape(clf_logits, [-1, inputs.shape[1]])

class ClfHead(keras.Model):
    """Classification Head for the transformer

    TODO: test this class."""
    def __init__(self, clf_token=40480, cfg=Args, n_class=10):
        super(ClfHead, self).__init__()
        self.n_embd = cfg.n_embd
        self.clf_token = clf_token
        self.dropout = keras.layers.Dropout(cfg.clf_pdrop)
        self.linear = keras.layers.Dense(n_class, input_shape=[cfg.n_embd],
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
            bias_initializer=keras.initializers.RandomNormal(stddev=1))

    def call(self, hidden, inputs):
        clf_h = tf.reshape(hidden, [-1, self.n_embd])
        flat = tf.reshape(inputs[..., 0], [-1])
        clf_h = clf_h[flat == self.clf_token, :]
        clf_h = tf.boolean_mask(clf_h, tf.equal(flat, self.clf_token))
        clf_h = self.dropout(clf_h)
        clf_logits = self.linear(clf_h)

        return clf_logits

class SimilarityHead(Model):
    """计算相似度的模型head"""
    def __init__(self, clf_token=40480, cfg=Args, n_class=10):
        super(SimilarityHead, self).__init__()
        self.n_embd = cfg.n_embd
        self.clf_token = clf_token
        self.dropout = keras.layers.Dropout(cfg.clf_pdrop)
        self.linear = keras.layers.Dense(n_class, input_shape=[cfg.n_embd],
                                         kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                                         bias_initializer=keras.initializers.RandomNormal(stddev=1))

    def call(self, hidden, inputs):
        sim_hidden = tf.reshape(hidden, [-1, self.n_embd])
        flat = tf.reshape(inputs[..., 0], [-1])
        sim_hidden = tf.boolean_mask(sim_hidden, tf.equal(flat, self.clf_token))
        sim_hidden = self.dropout(sim_hidden)
        sim_hidden = tf.reduce_sum(sim_hidden, axis=1)
        sim_logits = self.linear(sim_hidden)

        return sim_logits

class LMModel(Model):
    def __init__(self, cfg=Args, vocab=40990, n_ctx=512, retrun_probs=False):
        self.transformer = TransformerModel(cfg, vocab, n_ctx)
        self.lm_head = LMHead(self.transformer, cfg, trunc_and_shape=False)
        self.return_probs = retrun_probs
        if self.return_probs:
            self.pos_emb_mask = tf.zeros([1, 1, vocab])
            self.pos_emb_mask[:, :, -n_ctx:] = -1e12

    def call(self, inputs):
        hidden = self.transformer(inputs)
        lm_logits = self.lm_head(hidden)
        if self.return_probs:
            lm_logits = tf.nn.softmax(lm_logits + self.pos_emb_mask, -1)
        return lm_logits

class DoubleHeadModel(Model):
    """ Transformer with language model and task specific heads """

    def __init__(self, cfg=Args, clf_token=40480, task_head_type='multiple_choice', vocab=40990, n_ctx=512):
        super(DoubleHeadModel, self).__init__()
        self.transformer = TransformerModel(cfg, vocab=vocab, n_ctx=n_ctx)
        self.lm_head = LMHead(self.transformer, cfg)
        if isinstance(task_head_type, str):
            if task_head_type == 'multiple_choice':
                self.task_head = MultipleChoiceHead(clf_token, cfg)
            elif task_head_type == 'similarity':
                self.task_head = SimilarityHead(clf_token, cfg)
            elif task_head_type == 'inference':
                # the three classes correspond to entailment, contradiction and neutral.
                self.task_head = ClfHead(clf_token, cfg, 3)
            else:
                raise ValueError("task_head_type is expected to be 'multiple_choice' "
                                 "'similarity', 'inference' or ('classification', n_class) "
                                 f"got {task_head_type}.")
        elif isinstance(task_head_type, collections.abc.Sequence) and len(task_head_type) == 2 and \
                task_head_type[0] == 'classification':
            n_class = task_head_type[1]
            self.task_head = ClfHead(clf_token, cfg, n_class)
        else:
            raise ValueError("task_head_type is expected to be 'multiple_choice' "
                             "'similarity', 'inference' or ('classification', n_class) "
                             f"got {task_head_type}.")

    def call(self, x):
        h = self.transformer(x)
        lm_logits = self.lm_head(h)
        task_logits = self.task_head(h, x)

        return lm_logits, task_logits


if __name__ == '__main__':
    print(gelu(tf.constant(2.0)).numpy().item())
    print(swish(tf.constant(1.25)).numpy().item())