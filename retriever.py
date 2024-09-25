from FlagEmbedding import BGEM3FlagModel


class Retriever:
    def __init__(self, model_name, weights_for_different_modes=[1, 1, 1], device="cuda"):
        self.model = BGEM3FlagModel(
            model_name,
            device=device,
        )
        self.weights_for_different_modes = weights_for_different_modes

    def encode_documents(self, documents):
        self.documents = documents
        self.documents_embeddings = self.model.encode(
            documents,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )

    def encode_query(self, query):
        query_embedding = self.model.encode(
            query,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )
        return query_embedding

    def retrieve(self, query, top_k=5):
        if self.weights_for_different_modes[0] != 0:
            query_embedding = self.encode_query(query)
            dense_scores = (
                query_embedding["dense_vecs"]
                @ self.documents_embeddings["dense_vecs"].T
            )
            dense_scores = dense_scores.squeeze().tolist()
        else:
            dense_scores = [0] * len(self.documents)
        if self.weights_for_different_modes[1] != 0:
            colbert_scores = [
                self.model.colbert_score(
                    query_embedding["colbert_vecs"][0],
                    vectors,
                ).item()
                for vectors in self.documents_embeddings["colbert_vecs"]
            ]
        else:
            colbert_scores = [0] * len(self.documents)
        if self.weights_for_different_modes[2] != 0:
            lexical_scores = [
                self.model.compute_lexical_matching_score(
                    query_embedding["lexical_weights"][0],
                    doc,
                )
                for doc in self.documents_embeddings["lexical_weights"]
            ]
        else:
            lexical_scores = [0] * len(self.documents)

        aggregated_scores = [
            self.weights_for_different_modes[0] * dense_score
            + self.weights_for_different_modes[1] * colbert_score
            + self.weights_for_different_modes[2] * lexical_score
            for dense_score, colbert_score, lexical_score in zip(
                dense_scores, colbert_scores, lexical_scores
            )
        ]

        top_k_indices = sorted(
            range(len(aggregated_scores)),
            key=lambda i: aggregated_scores[i],
            reverse=True,
        )[:top_k]

        return top_k_indices