# 面试要点 Q&A

10 high-value questions an interviewer would ask about this project.
Practice each answer until you can say it in ~1 minute without reading.

---

## Q1: 什么是混合检索？为什么比纯语义检索好？

**Answer**:
混合检索结合了两种互补的检索方式：

1. **语义检索（Dense）**：将文本转成稠密向量，用余弦相似度衡量语义相近程度。擅长处理"同义不同词"的情况，比如"汽车"和"机动车"。

2. **关键词检索（BM25/Sparse）**：基于词频统计，擅长精确匹配专有名词、产品型号、数字等。

两者互补：语义检索解决同义词问题，BM25解决关键词精确匹配问题。用RRF算法融合两路结果，综合召回率更高。

在SuperMew里，`rag_utils.py` 的 `hybrid_retrieve` 就是这个思路，`MilvusManager.hybrid_retrieve()` 同时发两个 AnnSearchRequest。

---

## Q2: RRF算法原理是什么？为什么不直接加权平均分数？

**Answer**:
RRF（Reciprocal Rank Fusion）公式：

```
score(doc) = Σ  1 / (k + rank_i)
```

其中 k=60 是平滑系数，`rank_i` 是文档在第 i 路检索中的排名（从1开始）。

**为什么不加权平均分数**：
- 语义向量的余弦分数和BM25分数处于不同的数值空间，不能直接比较
- RRF只看排名，不看分数绝对值，天然避免了归一化问题
- k=60防止第1名独占太多分数，对排名靠后但多路都命中的文档更友好

这就是SuperMew里 `rag_utils.py` 注释里写的"RRF instead of weighted sum for cross-space robustness"。

---

## Q3: BM25和TF-IDF有什么区别？

**Answer**:
TF-IDF的问题是词频是线性的，一个词出现100次比出现10次权重高10倍，但实际意义并没有那么大。

BM25改进了两点：

1. **TF饱和（k1参数）**：`tf / (tf + k1)`，词频增加到一定程度后权重趋于饱和。k1=1.5是常用值。

2. **文档长度归一化（b参数）**：`b=0.75`，避免长文档因为包含更多词汇而获得不公平的高分。

公式：
```
BM25 = idf × (tf × (k1+1)) / (tf + k1 × (1 - b + b × |d|/avgdl))
```

SuperMew里用jieba分词后计算BM25，在Java版本里我用HanLP替代jieba。

---

## Q4: 层级分块和Auto-merge解决什么问题？你的参数是怎么设计的？

**Answer**:
这是精度和上下文的 trade-off 问题：

- 短 chunk（L3）：检索精度高，向量语义集中
- 长 chunk（L1）：上下文完整，但向量模糊、召回精度低

**Auto-merge 思路**：只存 L3 leaf chunk 到 Milvus 做检索，但如果同一段落里有 ≥2 个兄弟 L3 同时被命中，说明整段都相关，升级返回 L2 父块给 LLM。两轮迭代：L3→L2，L2→L1。

---

**亮点：参数设计经过踩坑和优化（这段最能体现思考深度）**

最初照搬平铺分块经验：L1=1200, L2=600, L3=300，overlap=50，结果暴露了三个问题：

1. **threshold 形同虚设**：2x 比例导致每个 L2 只有 2~3 个 L3 子块，threshold=2 几乎总触发，等于没有门槛。
2. **overlap 制造假相关**：相邻 L3 有 50 字重叠，向量高度相似，两块同时被检索命中 → 虚假触发升级 → 升级后内容大量重复。
3. **overlap 误用场景**：overlap 是平铺分块弥补边界截断的补丁，层级结构里父块已包含完整子块内容，overlap 无意义，反而是噪声来源。

对标 LlamaIndex 的 4x 比例（2048→512→128）后重新设计：

```
L1=1024字，L2=256字，L3=64字，overlap 全为 0
L3 使用标点感知语义切分（往前找。！？，），避免关键词截断
threshold=2 → 命中 ≥50% 子块才升级，含义清晰
```

这样 threshold=2 真正代表"整段高度相关"，auto-merge 的噪声和误触发大幅降低。

关键设计：L1/L2 不写入 Milvus（避免冗余向量），只存 JSON ParentChunkStore，升级时按需读取。

---

## Q5: HyDE是什么？什么时候用？

**Answer**:
HyDE = Hypothetical Document Embedding（假设文档嵌入）。

**思路**：直接用问题的向量去检索，问题和文档的向量空间有差距（问题是疑问句，文档是陈述句）。HyDE先让LLM根据问题生成一段"假设性答案"，用这段文本的向量去检索，向量空间更接近真实文档。

**什么时候用**：当初始检索结果被LLM评分为"不相关"（GradingService返回"no"）时，作为查询改写的一种策略。

SuperMew里在 `rag_utils.py` 的 `generate_hypothetical_document()` 实现，在 `rag_pipeline.py` 的 `rewrite_question` 节点里调用。

**注意**：HyDE会多一次LLM调用，有延迟成本，所以只在初检不合格时触发。

---

## Q6: 相关性评分（Grading）有什么用？整个流程是怎样的？

**Answer**:
Grading解决"召回了但不相关"的问题，流程如下：

```
检索 → 逐文档评分（LLM: yes/no）→
  yes: 进入生成
  no:  查询改写（step-back/HyDE）→ 重新检索 → 生成
```

评分用LLM structured output，输出 `{"score": "yes"}` 或 `{"score": "no"}`，简单可靠。

**实际价值**：
- 减少无关文档进入生成，降低hallucination
- 通过查询改写实现"多跳"检索能力

SuperMew里是 `rag_pipeline.py` 的 LangGraph 状态机：`grade_documents` → conditional edge → `rewrite_question` → `retrieve_expanded`。

---

## Q7: Milvus混合检索的具体实现？

**Answer**:
Milvus 2.4+ 原生支持混合检索，不需要在应用层合并：

1. **Schema**：collection里有两个向量字段：`dense_vector`（FloatVector, 1024维）和 `sparse_vector`（SparseFloatVector）

2. **两个AnnSearchRequest**：分别对dense和sparse字段做近似最近邻搜索，各取top-20

3. **RRFRanker**：Milvus内置的RRF融合，在数据库层合并结果

4. **filter_expr**：`chunk_level == 3` 只搜索leaf chunk，不搜索parent chunk

Java SDK：`HybridSearchParam` + `AnnSearchRequest` + `RRFRanker`。

优势：融合在数据库层完成，比应用层合并更高效，结果也更准确。

---

## Q8: 为什么只把L3写入Milvus，L1/L2存JSON？

**Answer**:
核心原因：**避免冗余向量干扰检索**。

L1 chunk包含L2，L2包含L3，内容高度重叠。如果三级都写入Milvus：
- 查一个问题会同时命中L1/L2/L3里的多个chunk，但它们表达的是同一段内容
- 向量空间里充斥着高度相似的向量，影响ANN索引质量
- 存储成本增加3倍

**设计决策**：只存最细粒度的L3（检索精度最高），需要更多上下文时在应用层从JSON store里取父chunk，这是auto-merge做的事。

这个设计在SuperMew里体现在 `milvus_writer.py` 只写L3，`parent_chunk_store.py` 存L1/L2。

---

## Q9: 流式输出是怎么实现的？前端怎么接收？

**Answer**:
**后端（Python/Java）**：
- LLM返回token流（streaming API）
- 用SSE（Server-Sent Events）协议推送给前端
- Content-Type: `text/event-stream`
- 每个token是一个SSE事件：`event: content\ndata: 你好\n\n`

**前端**：
- `fetch` + `ReadableStream` + `getReader()`
- 按行解析SSE数据
- 每收到一个token就追加到消息气泡里（打字机效果）

**SuperMew特色 — RAG步骤也走SSE**：
- 检索中/评分中/改写中这些中间步骤也作为 `rag_step` 类型的SSE事件推送
- 前端实现状态机：Idle → Thinking → RAG步骤 → 流式文本

Java版本用Spring WebFlux `Flux<ServerSentEvent<String>>`，比Python asyncio Queue更简洁。

---

## Q10: Spring AI Alibaba和LangChain有什么区别？为什么选Spring AI做Java迁移？

**Answer**:
| 维度 | LangChain (Python) | Spring AI Alibaba (Java) |
|---|---|---|
| 生态 | Python-first，AI原生 | Java/Spring生态，企业级 |
| 类型安全 | 动态类型，灵活但易出错 | 编译时类型检查，更可靠 |
| 依赖注入 | 手动组装 | Spring Bean管理，自动装配 |
| 流式支持 | asyncio + Queue (hack) | Reactor Flux，原生响应式 |
| 阿里云集成 | 需要手动封装 | Spring AI Alibaba原生支持DashScope |

**为什么选Spring AI Alibaba**：
1. 目标是Java后端简历项目，Spring生态更契合
2. Spring AI Alibaba针对通义千问/DashScope有一等公民支持
3. 响应式流式输出比SuperMew的跨线程asyncio方案更优雅
4. 企业面试更看重Spring生态经验

**SuperMew的价值**：作为参考实现理解核心算法（BM25、RRF、auto-merge），Java版本是重新实现而不是翻译。
