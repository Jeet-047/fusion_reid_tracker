# NOTE: The main tracking pipeline is implemented in main.py
# The following is pseudocode and not used in the actual project.
'''
for frame in video:
    # Detect
    boxes = yolo.detect(frame)

    # Crop & embed
    crops = [extract_box(frame, b) for b in boxes]
    embeds = model(crops)

    # Match to memory
    ids = match_embeddings(embeds, id_manager)

    # Assign new IDs if unmatched
    for i, id_ in enumerate(ids):
        if id_ == -1:
            ids[i] = id_manager.get_next_id()
        id_manager.update(ids[i], embeds[i])

    # Draw result
    draw_tracking(frame, boxes, ids)
'''
