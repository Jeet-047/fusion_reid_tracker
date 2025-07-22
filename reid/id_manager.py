class IDManager:
    def __init__(self):
        self.next_id = 1
        self.id_map = {}        # track_id -> embedding
        self.history = []       # embeddings for re-id

    def update(self, track_id, embedding):
        self.id_map[track_id] = embedding
        self.history.append((track_id, embedding))

    def get_next_id(self):
        id_ = self.next_id
        self.next_id += 1
        return id_
