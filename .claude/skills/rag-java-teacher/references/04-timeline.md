# 学习时间规划 (10 Days)

Assumes ~2-3 hours/day part-time. Adjust based on your pace.

---

## Day 1-2: Phase 1 — Basic RAG

**Daily target**: ~2-3h/day

**Day 1**:
- [ ] Create Spring Boot project (Spring Initializr)
- [ ] Add Spring AI Alibaba + Milvus dependencies to `pom.xml`
- [ ] Configure `application.yml` (DashScope API key, Milvus connection)
- [ ] Start Milvus with `docker compose up -d`
- [ ] Verify Milvus connection (Attu UI at `localhost:8000`)

**Day 2**:
- [ ] Implement PDF upload endpoint (`PagePdfDocumentReader` → `TokenTextSplitter` → `VectorStore.add()`)
- [ ] Implement `/chat` endpoint with `QuestionAnswerAdvisor`
- [ ] Implement `/chat/stream` SSE endpoint
- [ ] Test: upload a PDF → ask a question → get streaming answer

**Phase 1 Milestone**: Working Q&A demo. Show to someone who hasn't seen it — they should get an answer from the PDF.

---

## Day 3-5: Phase 2 — Hybrid Search

**Day 3**:
- [ ] Read `02-knowledge-points.md` sections: BM25, RRF
- [ ] Implement `BM25Indexer` class (theory → code)
- [ ] Choose tokenizer: HanLP or jieba4j — add to `pom.xml`
- [ ] Implement `BM25Service.search()`

**Day 4**:
- [ ] Implement `RRFMerger`
- [ ] Implement `HybridRetriever` combining dense + BM25
- [ ] Update upload endpoint to also index in BM25Service
- [ ] Test: query with a keyword (product name, number) — verify BM25 helps

**Day 5**:
- [ ] Implement `HybridRagAdvisor` (replace `QuestionAnswerAdvisor`)
- [ ] Add `/chat` filter: only retrieve `chunk_level == 3` leaves
- [ ] Compare: dense-only vs hybrid recall on 3 test questions
- [ ] Document comparison results (for resume)

**Phase 2 Milestone**: Measurably better keyword recall. "混合检索比纯语义提升约X%召回率" — write this number down.

---

## Day 6-8: Phase 3 — Advanced Features

**Day 6**:
- [ ] Read `02-knowledge-points.md` section: Hierarchical Chunking
- [ ] Implement `HierarchicalDocumentSplitter` (L1/L2/L3)
- [ ] Update upload endpoint to use hierarchical splitter
- [ ] Store L1/L2 in `ParentChunkStore`, L3 in Milvus

**Day 7**:
- [ ] Implement auto-merge logic in `HybridRetriever`
- [ ] Test auto-merge: upload long document, verify parent chunk returned when multiple siblings hit
- [ ] (Optional) Implement `GradingService` (binary relevance)

**Day 8**:
- [ ] (Optional) Implement `QueryRewriter` (step-back + HyDE)
- [ ] (Optional) Wire grading → rewrite → re-retrieve pipeline
- [ ] Polish: error handling, logging, meaningful responses when no context found

**Phase 3 Milestone**: Can demonstrate that for a multi-part document, the system returns the relevant parent section instead of fragmented leaf chunks.

---

## Day 9-10: Phase 4 — Polish + Resume

**Day 9**:
- [ ] Add session history (`MessageChatMemoryAdvisor` with `sessionId` param)
- [ ] Add `/sessions` endpoint (list, create, delete)
- [ ] Add document list/delete endpoints
- [ ] Frontend: copy SuperMew's `index.html`/`script.js`/`style.css` to `src/main/resources/static/`
- [ ] Update API base URL in frontend

**Day 10**:
- [ ] End-to-end demo run: upload PDF → ask questions → check streaming → check session memory
- [ ] Write resume bullet points (4-6 points, STAR format)
- [ ] Practice explaining each feature aloud (simulate interview)
- [ ] Record a 2-min demo video (optional but impressive)

**Final Milestone**: Can demo the full application and explain every technical decision in an interview.

---

## Checkpoint Questions (ask yourself each phase end)

**After Phase 1**: Can I explain what `QuestionAnswerAdvisor` does internally?
**After Phase 2**: Can I explain why RRF uses ranks instead of scores?
**After Phase 3**: Can I explain the trade-off between L3 precision and L1 context?
**After Phase 4**: Can I answer all 10 questions in `05-interview-points.md` without reading?

---

## If You Fall Behind

- **Skip Phase 3 advanced features** (grading/rewrite) — Phase 1+2 is already a strong demo
- **Use simple BM25** (no HanLP, just split on spaces) — works for demo purposes
- **Skip frontend** — terminal curl demo is fine for a backend role
- **Focus on explaining, not finishing** — partial implementation + deep understanding > full implementation + shallow understanding
