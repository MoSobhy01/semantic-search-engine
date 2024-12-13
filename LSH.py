import numpy as np
from sklearn.preprocessing import normalize
import heapq
from bitarray import bitarray
from sklearn.cluster import MiniBatchKMeans
import os
import gc
import json
import pickle

class LSH:
    def __init__(self, db, db_size, index_file_path = "lsh_index.npz", num_hashes = 4, num_tables=1, dim=70):
        self.index_file_path = index_file_path
        self.dim = dim
        self.db = db
        self.num_tables = num_tables
        self.hash_tables = {}
        self.projections = []
        self.num_hashes = num_hashes
        if(db_size):
          self.n_vectors = db_size
        else:
          self.n_vectors = db._get_num_records()
        self.n_clusters = int(np.sqrt(self.n_vectors))  # Rule of thumb for number of clusters
        if self.n_vectors > 15 * 10 ** 6:
            self.n_clusters = int(np.sqrt(self.n_vectors) * 5)



    def build_index(self, vectors):
        print("Building LSH index...")
        # Step 1: Normalize vectors to unit length (for cosine similarity)
        vectors = normalize(vectors, axis=1)

        # Step 2: Cluster the vectors using kmeans
        # Perform k-means clustering

        print("KMeans Training...")
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state = 42, n_init=10, batch_size=10000)
        print("KMeans Prediction...")
        cluster_labels = self.kmeans.fit_predict(vectors)

        # Group vectors by cluster
        clusters = {}
        for idx, label in enumerate(cluster_labels):
          if label not in clusters:
            clusters[label] = []
          clusters[label].append((idx, vectors[idx]))

        # Store Cluster centroids
        self.cluster_centroids = self.kmeans.cluster_centers_

        # Step 3: Generate random hyperplanes for hashing
        print("Generating random hyperplanes...")
        self.projections = np.random.randn(self.num_hashes * self.num_tables, self.dim)

        # Step 4: Apply LSH to each cluster
        print("Applying LSH to each cluster...")
        self.hash_tables = [{} for _ in range(self.n_clusters)]
        for label, cluster_vectors in clusters.items():
          self._build_index(cluster_vectors ,label)

        print("Indexing done.")
        del clusters
        del cluster_labels
        gc.collect()

    def _build_index(self, vectors, centroid_idx):
        # Initialize hash tables
        for table in range(self.num_tables):
            self.hash_tables[centroid_idx][table] = {}

        # Hash each vector into multiple hash tables
        for i, vector in vectors:
            hash_values = self._hash_vector(vector)
            for table in range(self.num_tables):
                hash_key = tuple(hash_values[table])
                # Append the vector number to the hash table bucket
                if hash_key not in self.hash_tables[centroid_idx][table]:
                    self.hash_tables[centroid_idx][table][hash_key] = []
                self.hash_tables[centroid_idx][table][hash_key].append(i)



    def _hash_vector(self, vector):
      # Ensure compatible sizes
      vector = np.array(vector).reshape(-1)
      projections = self.projections.reshape(self.num_tables, self.num_hashes, self.dim)

      # Perform Dot Product on vector against hyperplane normals
      hash_values = [bitarray((projections[table] @ vector > 0).tolist()) for table in range(self.num_tables)]
      return hash_values

    def search(self, query, top_k=10, n_clusters=13):
      if self.n_vectors > 15 * 10 ** 6:
        n_clusters = 3

      centroids = self.load_centroids()

      # Normalize the query vector
      query = query / np.linalg.norm(query)

      # Predict the cluster centroids for the query
      distances = np.linalg.norm(centroids - query, axis=1)
      del centroids
      nearest_clusters = np.argpartition(distances, n_clusters)[:n_clusters]
      del distances

      # load the projection vectors
      self.load_projections()

      # load the metadata from the disk
      metadata = self.load_tables_metadata()

      # Collect candidates from hash tables
      heap = []
      hash_values = self._hash_vector(query)
      for cluster_idx in nearest_clusters:
        hash_table = self.load_hash_table(cluster_idx, metadata)
        # Hash the query vector
        for table_idx in range(self.num_tables):
          table = hash_table[table_idx]
          hash_key = tuple(hash_values[table_idx])
          # Consider the exact hash key
          candidate_keys = [hash_key]
          # # Consider hash keys with Hamming distance of 1
          for bit_idx in range(len(hash_key)):
            modified_key = list(hash_key)
            modified_key[bit_idx] = not modified_key[bit_idx]
            candidate_keys.append(tuple(modified_key))

          cluster_candidates_rows =[]
          for candidate_key in candidate_keys:
            if candidate_key not in table:
              continue
            cluster_candidates_rows.extend(table[candidate_key])

          candidates_vectors = self.db.get_multiple_rows(cluster_candidates_rows)
          for candidate_id in candidates_vectors:
            vector = candidates_vectors[candidate_id]
            vector = vector / np.linalg.norm(vector)
            score = np.dot(query, vector)
            if len(heap) < top_k:
              heapq.heappush(heap, (score, candidate_id))
            elif score > heap[0][0]:
              heapq.heapreplace(heap, (score, candidate_id))
          del hash_table
          del candidates_vectors
          del cluster_candidates_rows
          gc.collect()

      candidates = heapq.nlargest(top_k, heap)
      top_k = [candidate_id for _, candidate_id in candidates]

      del candidates
      del heap
      del hash_values
      del metadata
      del self.projections
      del nearest_clusters
      gc.collect()
      return top_k


    def save(self):
        save_dir = f"{self.n_vectors // 10 ** 6}m"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        projections_path = os.path.join(save_dir, "lsh_projections.npy")
        centroids_path = os.path.join(save_dir, "kmeans_centroids.npy")

        hash_tables_path = os.path.join(save_dir, "lsh_hash_tables.pkl")
        cluster_offsets = {}

        with open(hash_tables_path, 'wb') as f:
            for cluster_id in range(len(self.hash_tables)):
              table = self.hash_tables[cluster_id]
              start_offset = f.tell()
              pickle.dump(table, f)
              end_offset = f.tell()
              cluster_offsets[cluster_id] = (start_offset, end_offset)

        offsets_path = os.path.join(save_dir, "cluster_offsets.json")
        with open(offsets_path, 'w') as f:
            json.dump(cluster_offsets, f)

        print("The size of metadatas is " + str(os.path.getsize(offsets_path) / (1024 * 1024)) + "MB")

        np.save(centroids_path, self.cluster_centroids)
        np.save(projections_path, self.projections)

        hash_size = os.path.getsize(hash_tables_path)

        centroids_size = os.path.getsize(centroids_path)
        projections_size = os.path.getsize(projections_path)
        total_size = hash_size + centroids_size + projections_size
        file_size_mb = total_size / (1024 * 1024)

        print("Hash Tables Size: " + str(hash_size / (1024 * 1024)) + "MB")
        print("Cluster Centroids Size: " + str(centroids_size / (1024 * 1024)) + "MB")
        print("Projections Size: " + str(projections_size / (1024 * 1024)) + "MB")
        print(f"Index File Size: {file_size_mb:.2f} MB")

    def load_tables_metadata(self):
        metadata_path = self.index_file_path + "/cluster_offsets.json"
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        return data

    def load_hash_table(self, cluster_id, metadata):
        hash_tables_path = self.index_file_path + "/lsh_hash_tables.pkl"
        with open(hash_tables_path, 'rb') as f:
            f.seek(metadata[str(cluster_id)][0])
            table = pickle.load(f)
        return table

    def load_centroids(self):
        centroids_path = self.index_file_path + "/kmeans_centroids.npy"
        data = np.load(centroids_path, allow_pickle=True)
        return data
    def load_projections(self):
      projections_path = self.index_file_path + "/lsh_projections.npy"
      data = np.load(projections_path, allow_pickle=True)
      self.projections = data
