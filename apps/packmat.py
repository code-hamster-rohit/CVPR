from ultralytics import YOLO
import cv2, os, time, numpy as np, traceback

class ObjectTracker:
    """
    A simple object tracker based on centroid proximity.
    Manages tracks with unique IDs and counting status.
    """
    def __init__(self, max_distance=50):
        self.tracks = {}
        self.next_id = 1
        self.max_distance = max_distance
        self.frame_count = 0

    def update_tracks(self, detected_objects, line_y, counter):
        """
        Updates object tracks based on new detections.
        Matches detections to existing tracks or creates new ones.
        Handles counting when objects cross the line.
        """
        self.frame_count += 1
        current_detections = {centroid: bbox for centroid, bbox in detected_objects}
        
        matched_track_ids = set()
        
        track_ids = list(self.tracks.keys())
        temp_detections = current_detections.copy()

        for obj_id in track_ids:
            track = self.tracks[obj_id]
            prev_centroid = track["centroid"]
            
            best_match = None
            min_dist = self.max_distance

            for centroid, bbox in temp_detections.items():
                if isinstance(centroid, (tuple, list)) and len(centroid) == 2 and \
                   isinstance(prev_centroid, (tuple, list)) and len(prev_centroid) == 2:
                   distance = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))
                   if distance < min_dist:
                       min_dist = distance
                       best_match = (centroid, bbox)

            if best_match:
                matched_centroid, matched_bbox = best_match

                prev_centroid_for_crossing_check = self.tracks[obj_id]["centroid"]
                self.tracks[obj_id]["centroid"] = matched_centroid
                self.tracks[obj_id]["bbox"] = matched_bbox
                self.tracks[obj_id]["last_seen"] = self.frame_count

                matched_track_ids.add(obj_id)

                if matched_centroid in temp_detections:
                    del temp_detections[matched_centroid]

                if not self.tracks[obj_id]["counted"] and \
                   self.crosses_line(matched_centroid, line_y, prev_centroid_for_crossing_check):
                    counter += 1
                    self.tracks[obj_id]["counted"] = True

        for centroid, bbox in temp_detections.items():
            new_id = self.next_id
            self.tracks[new_id] = {
                "centroid": centroid,
                "bbox": bbox,
                "counted": False,
                "last_seen": self.frame_count
            }
            self.next_id += 1

            if not self.tracks[new_id]["counted"] and self.crosses_line(centroid, line_y, None):
                counter += 1
                self.tracks[new_id]["counted"] = True

        max_unseen_frames = 30
        current_tracks = self.tracks.copy()
        for obj_id, track_data in current_tracks.items():
            if self.frame_count - track_data["last_seen"] > max_unseen_frames:
                print(f"Removing stale track ID: {obj_id}")
                if obj_id in self.tracks:
                     del self.tracks[obj_id]
        
        return counter

    @staticmethod
    def crosses_line(centroid, line_y, prev_centroid=None):
        """ Checks if an object crossed the horizontal line y=`line_y`. """
        if not isinstance(centroid, (tuple, list)) or len(centroid) != 2:
            return False
        
        cx, cy = centroid

        if prev_centroid is None:
             return abs(cy - line_y) <= 10
        
        if not isinstance(prev_centroid, (tuple, list)) or len(prev_centroid) != 2:
             return abs(cy - line_y) <= 5
             
        prev_cx, prev_cy = prev_centroid
        
        crossed = (prev_cy < line_y <= cy) or (prev_cy > line_y >= cy)
        
        if crossed and abs(cy - line_y) < 20:
            return True
            
        return False

class VideoProcessor:
    """
    Handles video reading, processing (detection, tracking), CLOCKWISE rotation, display, and writing.
    """
    def __init__(self, video_path, model_path, output_path, line_y_proportion=0.3, desired_fps=30.01, conf_threshold=0.65):
        self.cap = cv2.VideoCapture(video_path)
        self.model = YOLO(model_path)
        self.output_path = output_path
        self.desired_fps = desired_fps
        self.conf_threshold = conf_threshold

        if not self.cap.isOpened():
            raise Exception(f"Error: Could not open video file: {video_path}")

        self.encoded_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.encoded_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Encoded dimensions reported by OpenCV: {self.encoded_width}x{self.encoded_height}")

        if self.encoded_width == 0 or self.encoded_height == 0:
            self.cap.release()
            raise Exception(f"Error: Video file {video_path} reported zero width or height.")

        self.frame_width = self.encoded_height
        self.frame_height = self.encoded_width
        print(f"Target dimensions after CLOCKWISE rotation: {self.frame_width}x{self.frame_height}")

        self.line_y = int(self.frame_height * line_y_proportion)
        print(f"Line Y position (on target frame): {self.line_y}")
        self.frame_time_ms = int(1000 / desired_fps)

        if self.frame_width <= 0 or self.frame_height <= 0:
             self.cap.release()
             raise ValueError(f"Target frame dimensions are invalid: {self.frame_width}x{self.frame_height}")

        self.output_video = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*'mp4v'), desired_fps, (self.frame_width, self.frame_height)
        )
        if not self.output_video.isOpened():
             print(f"Error: Could not open VideoWriter for path: {output_path}")
             print(f"Using Codec: mp4v, FPS: {desired_fps}, Frame Size: ({self.frame_width}, {self.frame_height})")
             self.cap.release()

        self.counter = 0
        self.tracker = ObjectTracker(max_distance=75)

    def process_video(self):
        """ Reads, rotates (CLOCKWISE), processes, displays, and writes video frames. """
        if not self.cap.isOpened():
             print("Error: VideoCapture is not opened.")
             return

        frame_id = 0
        cv2.namedWindow("Video with Counting", cv2.WINDOW_NORMAL)

        while True:
            start_time = time.time()

            ret, frame = self.cap.read()
            if not ret:
                print("Video ended or cannot read frame.")
                break

            try:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            except cv2.error as e:
                 print(f"Error during cv2.rotate: {e}. Skipping frame.")
                 continue

            frame_id += 1
            process_frame = frame

            try:
                 results = self.model(process_frame, conf=self.conf_threshold, classes=[0], verbose=False)
            except Exception as e:
                 print(f"Error during model inference: {e}")
                 continue

            line_start = (0, min(max(0, self.line_y), self.frame_height - 1))
            line_end = (self.frame_width, min(max(0, self.line_y), self.frame_height - 1))
            cv2.line(frame, line_start, line_end, (255, 255, 255), 2)

            detected_objects = self.process_detections(results, frame)

            try:
                self.counter = self.tracker.update_tracks(detected_objects, self.line_y, self.counter)
            except Exception as e:
                 print(f"Error during track update: {e}")
                 traceback.print_exc()

            self.draw_tracks(frame)
            self.display_counter(frame)

            if self.output_video and self.output_video.isOpened():
                 try:
                      self.output_video.write(frame)
                 except cv2.error as e:
                      print(f"Error writing frame to video file: {e}")

            try:
                 cv2.imshow("Video with Counting", frame)
            except cv2.error as e:
                 print(f"Error displaying frame with cv2.imshow: {e}")

            elapsed_time = time.time() - start_time
            wait_time_ms = max(1, self.frame_time_ms - int(elapsed_time * 1000))

            key = cv2.waitKey(wait_time_ms) & 0xFF
            if key == 27:
                print("ESC key pressed. Exiting.")
                break
            try:
                 if cv2.getWindowProperty("Video with Counting", cv2.WND_PROP_VISIBLE) < 1:
                      print("Window closed by user. Exiting.")
                      break
            except cv2.error:
                 print("Window seems to be closed. Exiting.")
                 break

        self.cleanup()

    def process_detections(self, results, frame_to_draw_on):
        """ Extracts relevant detections (class 0) and prepares data for tracker. """
        detected_objects = []
        if results and hasattr(results[0], 'boxes') and results[0].boxes is not None:
            for result in results[0].boxes:
                 if hasattr(result, 'cls') and result.cls is not None and result.cls.numel() > 0 and \
                    hasattr(result, 'xyxy') and result.xyxy is not None and result.xyxy.numel() > 0:

                     cls = int(result.cls.cpu().item()) if result.cls.is_cuda else int(result.cls.item())

                     if cls == 0:
                         xyxy_tensor = result.xyxy[0]
                         xyxy = xyxy_tensor.cpu().tolist() if xyxy_tensor.is_cuda else xyxy_tensor.tolist()
                         x1, y1, x2, y2 = map(int, xyxy)

                         h, w = frame_to_draw_on.shape[:2]
                         x1, y1 = max(0, x1), max(0, y1)
                         x2, y2 = min(w - 1, x2), min(h - 1, y2)

                         if x1 < x2 and y1 < y2:
                             centroid = self.get_centroid(x1, y1, x2, y2)
                             detected_objects.append((centroid, (x1, y1, x2, y2)))
                             self.draw_detection(frame_to_draw_on, x1, y1, x2, y2)
        return detected_objects

    @staticmethod
    def get_centroid(x1, y1, x2, y2):
        """ Calculates the center point (centroid) of a bounding box. """
        return (x1 + x2) // 2, (y1 + y2) // 2

    @staticmethod
    def draw_detection(frame, x1, y1, x2, y2):
        """ Draws a bounding box for a detected object. """
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)

    def draw_tracks(self, frame):
         """ Draws tracked objects with their IDs and centroids. """
         for obj_id, track in self.tracker.tracks.items():
             if "centroid" in track and track["centroid"] is not None:
                 centroid = track["centroid"]
                 if isinstance(centroid, (tuple, list)) and len(centroid) == 2:
                     cv2.circle(frame, tuple(map(int, centroid)), 5, (0, 0, 255), -1)
                     text_pos = (int(centroid[0]) - 10, int(centroid[1]) - 15)
                     cv2.putText(frame, f"ID:{obj_id}", text_pos,
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    def display_counter(self, frame):
        """ Displays the current object count on the frame. """
        text_pos_y = int(self.frame_height * 0.05)
        text_pos_x = int(self.frame_width * 0.05)
        text_pos = (max(10, text_pos_x), max(30, text_pos_y))
        cv2.putText(frame, f"Counter: {self.counter}", text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

    def cleanup(self):
        """ Releases video capture and writer resources, destroys windows. """
        print("Releasing resources...")
        if hasattr(self, 'cap') and self.cap.isOpened():
             self.cap.release()
        if hasattr(self, 'output_video') and self.output_video.isOpened():
             self.output_video.release()
        cv2.destroyAllWindows()
        time.sleep(0.5)
        print("Cleanup finished.")

def run():
    """ Sets up paths and initiates video processing. """
    video_path = r'inputs\videos\video.mp4'
    model_path = r'model\jerrycan_bundle_detection.pt'
    output_path = r'counting-detected-video-rotated-clockwise.mp4'

    processor = None
    try:
        print("Initializing Video Processor...")
        processor = VideoProcessor(video_path, model_path, output_path,
                                   line_y_proportion=0.2,
                                   desired_fps=30.01,
                                   conf_threshold=0.65)
        print("Starting video processing...")
        processor.process_video()
        print("Video processing finished.")

    except Exception as e:
        print(f"\n--- An error occurred during processing ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print("\n--- Traceback ---")
        traceback.print_exc()
        print("-----------------\n")

    finally:
        if processor is not None:
            print("Running cleanup from finally block...")
            processor.cleanup()
        else:
             cv2.destroyAllWindows()
             time.sleep(0.5)
             print("Cleanup attempted for potentially uninitialized processor.")

    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Deleted {output_path}")