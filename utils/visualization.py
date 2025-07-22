import cv2

def draw_boxes(frame, boxes, ids):
    for box, pid in zip(boxes, ids):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f'ID {pid}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return frame
