import cv2
from ultralytics import YOLO

def process_video(video_path, output_path=None, model_path=None, class_names=None, confidence_threshold=0.5, max_duration=30, save_video=False):
    print('2')
    if class_names is None:
        class_names = ['gun']  # Default to 'gun' if no class names are provided

    cap = cv2.VideoCapture(video_path)
    print('3')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    else:
        out = None

    frame_count = 0
    max_frames = int(fps * max_duration)  # max_duration seconds of video

    yolo_model = YOLO(model_path)
    print('4')
    while True:
        ret, frame = cap.read()
        if not ret:
            
            break
        
        results = yolo_model(frame)
        
        for result in results:
            classes = result.names
            cls = result.boxes.cls
            conf = result.boxes.conf
            detections = result.boxes.xyxy

            for pos, detection in enumerate(detections):
                detected_class = classes[int(cls[pos])]
                confidence = conf[pos]
                if detected_class in class_names and confidence >= confidence_threshold:
                    print(f"Danger warning: {detected_class.capitalize()} detected")
                    xmin, ymin, xmax, ymax = detection
                    label = f"Warning: {'Weapon_detected'.capitalize()} detected {confidence:.2f}"
                    color = (0, 0, 255)
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                    cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if out:
            out.write(frame)
        
        cv2.imshow("Detected Objects", frame)
        cv2.resizeWindow("Detected Objects", 600, 800)
        frame_count += 1
        if frame_count >= max_frames or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    video_path = r'videos/video8.mp4'
    output_path = r'videos/output_videos/Detection_Output.mp4'
    model_path = r'./Models/weights/best.pt'
    print('1')
    class_names = ['gun']
    process_video(video_path, output_path, model_path, class_names=class_names, confidence_threshold=0.3, max_duration=100, save_video=True) 


#  git commit -m "Model Fine Tuned For Specific Gun detection"
